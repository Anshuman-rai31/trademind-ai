"""Backtest Execution Service for TradeMind (Phase 9).

Integrates the Phase 4 PortfolioSimulator, baseline strategies, risk manager,
and SQLite persistence with a bounded background worker for non-blocking asynchronous execution.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from flask import Flask

from ..db import db, safe_json_dumps, safe_json_loads
from ..models.backtest import BacktestRun, PortfolioSnapshotRecord, TradeRecord
from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..simulation.engine import PortfolioSimulator
from ..simulation.models import SimulationConfig
from ..simulation.strategies.buy_and_hold import BuyAndHoldStrategy
from ..simulation.strategies.ma_crossover import MACrossoverStrategy
from .market_data_service import MarketDataService

logger = logging.getLogger(__name__)

# Bounded in-process worker pool for background backtest execution
_executor: Optional[ThreadPoolExecutor] = None
_running_lock = threading.Lock()
_active_backtest_ids = set()


def get_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Retrieve or initialize the bounded background ThreadPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BacktestWorker")
    return _executor


class BacktestService:
    """Coordinates backtest submission, background execution, and persistence."""

    def __init__(self, data_service: Optional[MarketDataService] = None):
        self.data_service = data_service or MarketDataService()

    def create_and_queue_backtest(
        self,
        app: Flask,
        name: str,
        symbol: str,
        strategy_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
        parameters: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> BacktestRun:
        """Create a backtest record in SQLite and dispatch execution."""
        clean_strategy = strategy_type.strip().lower()
        allowed_strategies = {"buy_and_hold", "ma_crossover", "ppo"}
        if clean_strategy not in allowed_strategies:
            raise ValueError(
                f"Unsupported strategy_type: '{strategy_type}'. Supported: {sorted(allowed_strategies)}"
            )

        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be strictly positive, got {initial_capital}")

        if start_date and end_date and start_date >= end_date:
            raise ValueError(f"start_date ({start_date}) must be earlier than end_date ({end_date})")

        # Verify historical data availability
        df = self.data_service.get_historical_dataframe(symbol, start_date=start_date, end_date=end_date)
        if df is None or len(df) < 2:
            raise ValueError(
                f"Historical data unavailable or insufficient for symbol '{symbol}' "
                f"in range {start_date or 'earliest'} -> {end_date or 'latest'}."
            )

        # Record backtest in DB
        config_payload = {
            "parameters": parameters or {},
            "symbol": symbol,
            "strategy_type": clean_strategy,
            "start_date": start_date or (df["timestamp"].iloc[0].strftime("%Y-%m-%d")),
            "end_date": end_date or (df["timestamp"].iloc[-1].strftime("%Y-%m-%d")),
            "initial_capital": initial_capital,
        }

        backtest = BacktestRun(
            name=name.strip(),
            symbol=symbol.strip().upper(),
            strategy_type=clean_strategy,
            start_date=config_payload["start_date"],
            end_date=config_payload["end_date"],
            initial_capital=float(initial_capital),
            status="queued",
            configuration_json=safe_json_dumps(config_payload),
        )
        db.session.add(backtest)
        db.session.commit()

        backtest_id = backtest.id

        if sync or app.config.get("TESTING", False):
            # Synchronous execution for deterministic tests or explicit sync flag
            self._execute_backtest_task(backtest_id, app)
            db.session.refresh(backtest)
            return backtest
        else:
            # Bounded asynchronous background execution
            executor = get_executor(app.config.get("MAX_BACKTEST_WORKERS", 2))
            with _running_lock:
                _active_backtest_ids.add(backtest_id)
            executor.submit(self._execute_backtest_task, backtest_id, app)
            return backtest

    def _execute_backtest_task(self, backtest_id: int, app: Flask) -> None:
        """Worker task executing simulation, updating status, and writing audit records."""
        with app.app_context():
            backtest: Optional[BacktestRun] = db.session.get(BacktestRun, backtest_id)
            if not backtest:
                logger.error("Backtest ID %d not found for execution.", backtest_id)
                with _running_lock:
                    _active_backtest_ids.discard(backtest_id)
                return

            try:
                backtest.status = "running"
                db.session.commit()

                cfg_data = safe_json_loads(backtest.configuration_json, {})
                params = cfg_data.get("parameters", {})
                sym = backtest.symbol

                # 1. Load data
                df = self.data_service.get_historical_dataframe(
                    sym,
                    start_date=backtest.start_date,
                    end_date=backtest.end_date,
                )
                if df is None or len(df) < 2:
                    raise ValueError(f"Insufficient historical data for simulation of {sym}.")

                # 2. Strategy instantiation
                strategy_type = backtest.strategy_type
                if strategy_type == "buy_and_hold":
                    alloc = float(params.get("allocation_pct", 0.98))
                    strategy = BuyAndHoldStrategy(target_symbol=sym, allocation_pct=alloc)
                elif strategy_type == "ma_crossover":
                    fast_w = int(params.get("fast_window", 10))
                    slow_w = int(params.get("slow_window", 50))
                    ma_type = str(params.get("ma_type", "SMA"))
                    alloc = float(params.get("allocation_pct", 0.95))
                    strategy = MACrossoverStrategy(
                        fast_window=fast_w,
                        slow_window=slow_w,
                        ma_type=ma_type,
                        target_symbol=sym,
                        allocation_pct=alloc,
                    )
                elif strategy_type == "ppo":
                    # PPO agent simulation requires pre-trained model artifact
                    raise ValueError(
                        "PPO simulation via backtest engine requires a saved PPO artifact. "
                        "Train an agent first before running PPO backtests."
                    )
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")

                # 3. Simulator & Risk setup
                sim_config = SimulationConfig(
                    initial_cash=backtest.initial_capital,
                    fee_rate=float(params.get("fee_rate", 0.0003)),
                    slippage_rate=float(params.get("slippage_rate", 0.0005)),
                    min_fee=float(params.get("min_fee", 0.0)),
                )

                risk_cfg = RiskConfig(
                    max_position_exposure=float(params.get("max_position_exposure", 1.0)),
                    max_drawdown_limit=float(params.get("max_drawdown_limit", 0.15)),
                    max_daily_loss_limit=float(params.get("max_daily_loss_limit", 0.05)),
                )
                risk_mgr = RiskManager(config=risk_cfg)

                simulator = PortfolioSimulator(config=sim_config, risk_manager=risk_mgr)

                # 4. Execute Backtest
                result = simulator.run(data=df, strategy=strategy)

                # 5. Persist Results & Audit Trails
                metrics_dict = result.metrics.to_dict()
                backtest.metrics_json = safe_json_dumps(metrics_dict)
                backtest.status = "completed"
                backtest.completed_at = datetime.now(timezone.utc)
                backtest.error_message = None

                # Persist Daily Valuation Snapshots
                if result.portfolio and result.portfolio.snapshots:
                    for s in result.portfolio.snapshots:
                        snap = PortfolioSnapshotRecord(
                            backtest_id=backtest.id,
                            timestamp=str(s.timestamp),
                            cash=float(s.cash),
                            holdings_value=float(s.holdings_value),
                            portfolio_value=float(s.total_value),
                            drawdown=float(s.drawdown),
                        )
                        db.session.add(snap)

                # Persist Executed Trades
                if result.trades_df is not None and not result.trades_df.empty:
                    for _, t_row in result.trades_df.iterrows():
                        side_str = str(t_row.get("side", ""))
                        if "BUY" in side_str:
                            action = "BUY"
                        elif "SELL" in side_str:
                            action = "SELL"
                        else:
                            action = side_str

                        trade = TradeRecord(
                            backtest_id=backtest.id,
                            symbol=str(t_row.get("symbol", sym)),
                            timestamp=str(t_row.get("timestamp", "")),
                            action=action,
                            quantity=float(t_row.get("quantity", 0.0)),
                            price=float(t_row.get("execution_price", t_row.get("price", 0.0))),
                            transaction_cost=float(t_row.get("fee", 0.0)) + float(t_row.get("slippage_cost", 0.0)),
                            realized_pnl=float(t_row.get("realized_pnl")) if pd.notnull(t_row.get("realized_pnl")) else None,
                        )
                        db.session.add(trade)

                db.session.commit()
                logger.info("Backtest ID %d (%s) completed successfully.", backtest.id, backtest.name)

            except Exception as exc:
                db.session.rollback()
                logger.exception("Backtest ID %d failed: %s", backtest_id, exc)
                backtest = db.session.get(BacktestRun, backtest_id)
                if backtest:
                    backtest.status = "failed"
                    backtest.error_message = str(exc)
                    backtest.completed_at = datetime.now(timezone.utc)
                    db.session.commit()

            finally:
                with _running_lock:
                    _active_backtest_ids.discard(backtest_id)
