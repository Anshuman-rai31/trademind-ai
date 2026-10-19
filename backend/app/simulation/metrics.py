"""TradeMind Phase 4 Performance Metrics Engine.

Calculates institutional quantitative risk, return, and trade execution metrics.
Guarantees numerical stability and gracefully handles zero-volatility edge cases,
empty periods, and insufficient historical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .models import PortfolioSnapshot, Trade


@dataclass
class PerformanceMetrics:
    """Institutional performance and risk summary.
    
    Attributes:
        initial_value: Starting portfolio equity (INR).
        final_value: Ending portfolio equity (INR).
        total_return: Net percentage portfolio return over the full backtest.
        cagr: Compound Annual Growth Rate (annualized geometric return).
        annualized_volatility: Sample standard deviation of daily returns scaled by sqrt(252).
        sharpe_ratio: Annualized risk-adjusted excess return over risk-free rate.
        sortino_ratio: Annualized return over downside semi-deviation.
        max_drawdown: Maximum peak-to-trough decline in portfolio equity.
        max_drawdown_duration_bars: Longest sequence of consecutive bars in drawdown.
        total_trades: Total number of executed transactions (buys + sells).
        buy_trades: Number of buy executions.
        sell_trades: Number of sell executions.
        winning_trades: Number of profitable closed round-trip trades.
        losing_trades: Number of unprofitable closed round-trip trades.
        win_rate: Percentage of profitable closed trades (0.0 to 1.0).
        profit_factor: Gross profits divided by gross losses on closed trades.
        total_fees: Cumulative brokerage and transaction fees paid (INR).
        total_slippage_cost: Cumulative slippage cost penalty incurred (INR).
        turnover: Total traded turnover divided by twice the average portfolio equity.
        risk_free_rate_used: Annualized risk-free rate assumed for Sharpe calculation.
        benchmark_total_return: Benchmark asset total return over same horizon (if provided).
        benchmark_cagr: Benchmark annualized return (if provided).
        total_bars: Number of daily bars evaluated in the backtest.
    """
    initial_value: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    turnover: float = 0.0
    risk_free_rate_used: float = 0.05
    benchmark_total_return: Optional[float] = None
    benchmark_cagr: Optional[float] = None
    total_bars: int = 0

    def to_dict(self) -> Dict[str, Union[float, int, Optional[float]]]:
        """Convert metrics to a clean serializable dictionary."""
        return {
            "initial_value": round(self.initial_value, 2),
            "final_value": round(self.final_value, 2),
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "annualized_volatility": round(self.annualized_volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "total_trades": self.total_trades,
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_fees": round(self.total_fees, 2),
            "total_slippage_cost": round(self.total_slippage_cost, 2),
            "turnover": round(self.turnover, 4),
            "risk_free_rate_used": self.risk_free_rate_used,
            "benchmark_total_return": round(self.benchmark_total_return, 4) if self.benchmark_total_return is not None else None,
            "benchmark_cagr": round(self.benchmark_cagr, 4) if self.benchmark_cagr is not None else None,
            "total_bars": self.total_bars,
        }


def calculate_metrics(
    snapshots: Sequence[PortfolioSnapshot],
    trades: Sequence[Trade],
    risk_free_rate: float = 0.05,
    benchmark_prices: Optional[Sequence[float]] = None,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Calculate institutional performance and risk metrics from backtest results.
    
    Args:
        snapshots: Chronological sequence of PortfolioSnapshot instances.
        trades: Chronological sequence of executed Trade instances.
        risk_free_rate: Annualized risk-free rate assumption (default 5.0% = 0.05).
        benchmark_prices: Optional series of benchmark asset prices matching snapshot bars.
        periods_per_year: Trading sessions per year (252 for NSE equities).
        
    Returns:
        Populated PerformanceMetrics instance.
    """
    n_bars = len(snapshots)
    if n_bars == 0:
        return PerformanceMetrics(risk_free_rate_used=risk_free_rate, total_bars=0)

    initial_value = snapshots[0].cash if n_bars == 1 else snapshots[0].total_value
    final_value = snapshots[-1].total_value
    total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0

    # Extract equity curve and daily returns
    equity_series = pd.Series([s.total_value for s in snapshots], dtype=float)
    daily_returns = equity_series.pct_change().dropna()

    # CAGR calculation
    cagr = 0.0
    if n_bars > 1 and initial_value > 0 and final_value > 0:
        years = n_bars / periods_per_year
        if years > 0:
            cagr = float((final_value / initial_value) ** (1.0 / years) - 1.0)

    # Annualized Volatility
    ann_vol = 0.0
    if len(daily_returns) > 1:
        daily_std = float(daily_returns.std(ddof=1))
        if not np.isnan(daily_std) and daily_std > 0:
            ann_vol = daily_std * np.sqrt(periods_per_year)

    # Sharpe Ratio
    # Excess return over daily risk-free rate
    sharpe = 0.0
    if ann_vol > 1e-9:
        sharpe = float((cagr - risk_free_rate) / ann_vol)

    # Sortino Ratio (Downside semi-deviation)
    sortino = 0.0
    if ann_vol > 1e-9:
        daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        downside_returns = daily_returns[daily_returns < daily_rf] - daily_rf
        if len(downside_returns) > 0:
            downside_std = float(np.sqrt(np.mean(downside_returns ** 2))) * np.sqrt(periods_per_year)
            if downside_std > 1e-9:
                sortino = float((cagr - risk_free_rate) / downside_std)

    # Maximum Drawdown and Max Drawdown Duration
    running_max = equity_series.cummax()
    drawdowns = (running_max - equity_series) / running_max
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Drawdown duration in bars
    max_dd_duration = 0
    current_dd_duration = 0
    for dd in drawdowns:
        if dd > 1e-6:
            current_dd_duration += 1
            if current_dd_duration > max_dd_duration:
                max_dd_duration = current_dd_duration
        else:
            current_dd_duration = 0

    # Trade statistics
    total_trades = len(trades)
    buy_trades = sum(1 for t in trades if t.side.value == "BUY")
    sell_trades = sum(1 for t in trades if t.side.value == "SELL")

    # Realized PnL analysis on closed round-trip positions (SELL trades)
    closed_trades = [t for t in trades if t.realized_pnl is not None]
    winning_trades = sum(1 for t in closed_trades if t.realized_pnl > 0)
    losing_trades = sum(1 for t in closed_trades if t.realized_pnl < 0)
    win_rate = float(winning_trades / len(closed_trades)) if closed_trades else 0.0

    gross_profit = sum(t.realized_pnl for t in closed_trades if t.realized_pnl > 0)
    gross_loss = abs(sum(t.realized_pnl for t in closed_trades if t.realized_pnl < 0))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Transaction Costs
    total_fees = sum(t.fee for t in trades)
    total_slippage = sum(t.slippage * t.quantity for t in trades)

    # Turnover: total traded value / (2 * average portfolio equity)
    total_traded_value = sum(t.execution_price * t.quantity for t in trades)
    avg_portfolio_value = float(equity_series.mean()) if len(equity_series) > 0 else initial_value
    turnover = float(total_traded_value / (2.0 * avg_portfolio_value)) if avg_portfolio_value > 0 else 0.0

    # Benchmark metrics
    bench_total_ret = None
    bench_cagr = None
    if benchmark_prices is not None and len(benchmark_prices) >= 2:
        bp_start = float(benchmark_prices[0])
        bp_end = float(benchmark_prices[-1])
        if bp_start > 0:
            bench_total_ret = float((bp_end - bp_start) / bp_start)
            years = len(benchmark_prices) / periods_per_year
            if years > 0 and bp_end > 0:
                bench_cagr = float((bp_end / bp_start) ** (1.0 / years) - 1.0)

    return PerformanceMetrics(
        initial_value=float(initial_value),
        final_value=float(final_value),
        total_return=float(total_return),
        cagr=float(cagr),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        max_drawdown_duration_bars=int(max_dd_duration),
        total_trades=int(total_trades),
        buy_trades=int(buy_trades),
        sell_trades=int(sell_trades),
        winning_trades=int(winning_trades),
        losing_trades=int(losing_trades),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        total_fees=float(total_fees),
        total_slippage_cost=float(total_slippage),
        turnover=float(turnover),
        risk_free_rate_used=float(risk_free_rate),
        benchmark_total_return=bench_total_ret,
        benchmark_cagr=bench_cagr,
        total_bars=int(n_bars),
    )
