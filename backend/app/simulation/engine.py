"""TradeMind Phase 4 Backtesting and Simulation Engine.

Orchestrates deterministic portfolio simulation, order matching, fee & slippage
accounting, daily mark-to-market valuation, and institutional performance measurement.

Guarantees zero-lookahead bias: signals generated at bar t are queued and
executed at the next available trading opportunity (bar t+1 open).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, calculate_metrics
from .models import (
    Order,
    OrderStatus,
    PortfolioSnapshot,
    SimulationConfig,
    Trade,
)
from .portfolio import Portfolio
from .strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Comprehensive container for backtest results and audit trails.
    
    Attributes:
        strategy_name: Name of the executed strategy.
        config: SimulationConfig used.
        portfolio: Final Portfolio state ledger.
        metrics: Populated PerformanceMetrics instance.
        snapshots_df: Daily portfolio valuation DataFrame.
        trades_df: Executed trades audit DataFrame.
        orders_df: All order requests audit DataFrame.
    """
    strategy_name: str
    config: SimulationConfig
    portfolio: Portfolio
    metrics: PerformanceMetrics
    snapshots_df: pd.DataFrame
    trades_df: pd.DataFrame
    orders_df: pd.DataFrame
    risk_interventions: List[Any] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted textual summary of backtest performance."""
        m = self.metrics
        lines = [
            "=" * 64,
            f" TradeMind Backtest Summary: {self.strategy_name}",
            "=" * 64,
            f" Initial Capital      : INR {m.initial_value:,.2f}",
            f" Final Portfolio NAV  : INR {m.final_value:,.2f}",
            f" Net Total Return     : {m.total_return * 100:.2f}%",
            f" Annualized (CAGR)    : {m.cagr * 100:.2f}%",
            f" Annualized Volatility: {m.annualized_volatility * 100:.2f}%",
            f" Sharpe Ratio (rf={m.risk_free_rate_used * 100:.1f}%): {m.sharpe_ratio:.2f}",
            f" Sortino Ratio        : {m.sortino_ratio:.2f}",
            f" Maximum Drawdown     : {m.max_drawdown * 100:.2f}% (Duration: {m.max_drawdown_duration_bars} bars)",
            "-" * 64,
            f" Total Trades Filled  : {m.total_trades} (Buys: {m.buy_trades}, Sells: {m.sell_trades})",
            f" Win Rate             : {m.win_rate * 100:.2f}% (Wins: {m.winning_trades}, Losses: {m.losing_trades})",
            f" Total Fees Paid      : INR {m.total_fees:,.2f}",
            f" Total Slippage Cost  : INR {m.total_slippage_cost:,.2f}",
            f" Portfolio Turnover   : {m.turnover:.2f}x",
            f" Total Bars Simulated : {m.total_bars}",
        ]
        if m.benchmark_total_return is not None:
            lines.append(f" Benchmark Return     : {m.benchmark_total_return * 100:.2f}%")
        lines.append("=" * 64)
        return "\n".join(lines)


class PortfolioSimulator:
    """Institutional quantitative portfolio backtester.
    
    Supports single-asset or multi-asset equities, realistic execution
    slippage, configurable brokerage commissions, market holiday gap handling,
    and rigorous no-lookahead signal dispatching.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        """Initialize simulator with configuration parameters.
        
        Args:
            config: Optional SimulationConfig. If None, default settings are used.
            risk_manager: Optional Phase 7 RiskManager instance to evaluate pre-trade risks.
        """
        self.config: SimulationConfig = config or SimulationConfig()
        self.risk_manager = risk_manager

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy: BaseStrategy,
        benchmark_symbol: Optional[str] = None,
    ) -> SimulationResult:
        """Run backtest simulation for a strategy over historical data.
        
        Args:
            data: Standardized OHLCV DataFrame (with 'symbol' column or single asset)
                  or Dict mapping symbol -> OHLCV DataFrame.
            strategy: BaseStrategy instance generating trading signals.
            benchmark_symbol: Optional symbol to extract for benchmark comparison.
            
        Returns:
            SimulationResult containing ledger, metrics, and audit tables.
        """
        # 1. Normalize and structure input data
        normalized_data = self._prepare_data(data)
        symbols = list(normalized_data.keys())

        if not symbols:
            raise ValueError("Simulation data contains no valid symbols or records.")

        # 2. Build aligned chronological calendar
        all_timestamps = self._build_unified_calendar(normalized_data)
        if len(all_timestamps) == 0:
            raise ValueError("Simulation data contains no timestamps.")

        # 3. Initialize portfolio, strategy, and risk manager
        portfolio = Portfolio(initial_cash=self.config.initial_cash)
        strategy.initialize(config=self.config, symbols=symbols)
        if self.risk_manager:
            self.risk_manager.reset(self.config.initial_cash)

        pending_orders: List[Order] = []
        last_known_prices: Dict[str, float] = {}

        # 4. Chronological Simulation Loop
        for bar_idx, current_ts in enumerate(all_timestamps):
            # A. Execute pending orders from bar t-1 at current bar t execution price (e.g. Open)
            if pending_orders:
                orders_to_process = list(pending_orders)
                pending_orders.clear()

                for order in orders_to_process:
                    sym = order.symbol
                    if sym in normalized_data and current_ts in normalized_data[sym].index:
                        bar = normalized_data[sym].loc[current_ts]
                        exec_col = self.config.execution_price_col
                        base_price = float(bar[exec_col]) if exec_col in bar else float(bar["open"])

                        if base_price > 0 and not np.isnan(base_price):
                            approved_order: Optional[Order] = order
                            risk_interv = None
                            if self.risk_manager:
                                approved_order, risk_interv = self.risk_manager.evaluate_proposed_order(
                                    order=order,
                                    portfolio=portfolio,
                                    execution_price=base_price,
                                    timestamp=current_ts,
                                    step=bar_idx,
                                )
                            if approved_order is not None:
                                portfolio.execute_order(
                                    order=approved_order,
                                    base_price=base_price,
                                    timestamp=current_ts,
                                    config=self.config,
                                )
                            else:
                                order.status = OrderStatus.REJECTED
                                order.reason = risk_interv.details if risk_interv else "Rejected by risk manager"
                                portfolio.order_history.append(order)
                        else:
                            order.status = OrderStatus.REJECTED
                            order.reason = f"Invalid {exec_col} price on {current_ts}: {base_price}"
                            portfolio.order_history.append(order)
                    else:
                        # Asset did not trade on current timestamp (holiday, suspension, etc.)
                        # Keep order pending for next open opportunity
                        pending_orders.append(order)

            # B. Update mark-to-market prices for all assets
            current_prices: Dict[str, float] = {}
            for sym in symbols:
                sym_df = normalized_data[sym]
                if current_ts in sym_df.index:
                    val_col = self.config.valuation_price_col
                    val_price = float(sym_df.loc[current_ts, val_col]) if val_col in sym_df.columns else float(sym_df.loc[current_ts, "close"])
                    if val_price > 0 and not np.isnan(val_price):
                        last_known_prices[sym] = val_price
                # Carry forward last known price if missing on this session
                current_prices[sym] = last_known_prices.get(sym, 0.0)

            # C. Record daily portfolio valuation snapshot (End-of-day mark-to-market)
            snapshot = portfolio.record_snapshot(timestamp=current_ts, current_prices=current_prices)
            if self.risk_manager:
                self.risk_manager.update_portfolio_valuation(
                    timestamp=current_ts,
                    current_nav=snapshot.total_value,
                    step=bar_idx,
                )

            # D. Generate trading signals based strictly on data up to bar t (<= t)
            # Orders are queued for execution at bar t+1
            if bar_idx < len(all_timestamps) - 1:
                # Slice each symbol's data up to current timestamp
                historical_slices: Dict[str, pd.DataFrame] = {}
                for sym, sym_df in normalized_data.items():
                    sub_slice = sym_df.loc[sym_df.index <= current_ts]
                    if not sub_slice.empty:
                        historical_slices[sym] = sub_slice

                new_orders = strategy.generate_orders(
                    bar_idx=bar_idx,
                    timestamp=current_ts,
                    historical_data=historical_slices,
                    portfolio=portfolio,
                )
                if new_orders:
                    pending_orders.extend(new_orders)

        # 5. Build Benchmark series if requested
        benchmark_prices = None
        bench_sym = benchmark_symbol or symbols[0]
        if bench_sym in normalized_data:
            b_df = normalized_data[bench_sym]
            bench_prices = []
            for ts in all_timestamps:
                if ts in b_df.index:
                    bench_prices.append(float(b_df.loc[ts, "close"]))
                elif bench_prices:
                    bench_prices.append(bench_prices[-1])
                else:
                    bench_prices.append(0.0)
            benchmark_prices = bench_prices

        # 6. Calculate institutional risk and return metrics
        metrics = calculate_metrics(
            snapshots=portfolio.snapshots,
            trades=portfolio.trades,
            risk_free_rate=self.config.risk_free_rate,
            benchmark_prices=benchmark_prices,
        )

        # 7. Build audit DataFrames
        snapshots_df = self._snapshots_to_df(portfolio.snapshots)
        trades_df = self._trades_to_df(portfolio.trades)
        orders_df = self._orders_to_df(portfolio.order_history)

        return SimulationResult(
            strategy_name=strategy.name,
            config=self.config,
            portfolio=portfolio,
            metrics=metrics,
            snapshots_df=snapshots_df,
            trades_df=trades_df,
            orders_df=orders_df,
            risk_interventions=list(self.risk_manager.interventions) if self.risk_manager else [],
        )

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """Normalize and index input data by symbol and timestamp."""
        result: Dict[str, pd.DataFrame] = {}

        if isinstance(data, dict):
            for sym, df in data.items():
                prepared = self._normalize_single_df(df, default_symbol=sym)
                if not prepared.empty:
                    result[sym] = prepared
        elif isinstance(data, pd.DataFrame):
            if "symbol" in data.columns and data["symbol"].nunique() > 1:
                for sym, group in data.groupby("symbol"):
                    sym_str = str(sym)
                    prepared = self._normalize_single_df(group, default_symbol=sym_str)
                    if not prepared.empty:
                        result[sym_str] = prepared
            else:
                default_sym = str(data["symbol"].iloc[0]) if "symbol" in data.columns else "ASSET"
                prepared = self._normalize_single_df(data, default_symbol=default_sym)
                if not prepared.empty:
                    result[default_sym] = prepared
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return result

    @staticmethod
    def _normalize_single_df(df: pd.DataFrame, default_symbol: str) -> pd.DataFrame:
        """Ensure standard datetime index and canonical lowercase column names."""
        clean = df.copy()

        # Lowercase column names
        clean.columns = [c.lower() for c in clean.columns]

        # Extract timestamp index
        if "timestamp" in clean.columns:
            clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
            clean = clean.sort_values("timestamp")
            clean = clean.set_index("timestamp")
        elif not isinstance(clean.index, pd.DatetimeIndex):
            clean.index = pd.to_datetime(clean.index, utc=True)

        if "symbol" not in clean.columns:
            clean["symbol"] = default_symbol

        return clean

    @staticmethod
    def _build_unified_calendar(data_map: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Create a sorted, unique union of all session timestamps across assets."""
        all_indices = [df.index for df in data_map.values() if isinstance(df.index, pd.DatetimeIndex)]
        if not all_indices:
            return pd.DatetimeIndex([])
        unified = all_indices[0]
        for idx in all_indices[1:]:
            unified = unified.union(idx)
        return unified.sort_values()

    @staticmethod
    def _snapshots_to_df(snapshots: List[PortfolioSnapshot]) -> pd.DataFrame:
        """Convert list of PortfolioSnapshot objects to a DataFrame."""
        if not snapshots:
            return pd.DataFrame()
        records = [
            {
                "timestamp": s.timestamp,
                "cash": s.cash,
                "holdings_value": s.holdings_value,
                "total_value": s.total_value,
                "daily_return": s.daily_return,
                "cumulative_return": s.cumulative_return,
                "drawdown": s.drawdown,
            }
            for s in snapshots
        ]
        return pd.DataFrame(records).set_index("timestamp")

    @staticmethod
    def _trades_to_df(trades: List[Trade]) -> pd.DataFrame:
        """Convert list of Trade objects to a DataFrame."""
        if not trades:
            return pd.DataFrame(
                columns=[
                    "trade_id", "order_id", "symbol", "side", "quantity",
                    "base_price", "execution_price", "slippage", "fee",
                    "net_cost", "realized_pnl", "timestamp"
                ]
            )
        records = [
            {
                "trade_id": t.trade_id,
                "order_id": t.order_id,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "base_price": t.base_price,
                "execution_price": t.execution_price,
                "slippage": t.slippage,
                "fee": t.fee,
                "net_cost": t.net_cost,
                "realized_pnl": t.realized_pnl,
                "timestamp": t.timestamp,
            }
            for t in trades
        ]
        return pd.DataFrame(records)

    @staticmethod
    def _orders_to_df(orders: List[Order]) -> pd.DataFrame:
        """Convert list of Order objects to a DataFrame."""
        if not orders:
            return pd.DataFrame(
                columns=[
                    "order_id", "symbol", "side", "quantity", "status",
                    "created_at", "executed_at", "filled_price", "fee",
                    "slippage", "reason"
                ]
            )
        records = [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": o.quantity,
                "status": o.status.value,
                "created_at": o.created_at,
                "executed_at": o.executed_at,
                "filled_price": o.filled_price,
                "fee": o.fee,
                "slippage": o.slippage,
                "reason": o.reason,
            }
            for o in orders
        ]
        return pd.DataFrame(records)
