"""TradeMind Phase 4 Simulation Reporting & Diagnostics.

Formats backtest executions, benchmark comparisons, trade logs, and metrics
into human-readable tables and machine-readable audit dictionaries.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

from .engine import SimulationResult


def format_backtest_summary(result: SimulationResult) -> str:
    """Format a detailed ASCII report of backtest results."""
    return result.summary()


def compare_strategies(results: List[SimulationResult]) -> pd.DataFrame:
    """Tabulate and compare key metrics across multiple strategy runs.
    
    Args:
        results: List of SimulationResult instances to compare.
        
    Returns:
        DataFrame with strategies as columns and metrics as index rows.
    """
    data: Dict[str, Dict[str, str]] = {}
    for res in results:
        m = res.metrics
        data[res.strategy_name] = {
            "Initial Capital (INR)": f"{m.initial_value:,.2f}",
            "Final NAV (INR)": f"{m.final_value:,.2f}",
            "Total Return": f"{m.total_return * 100:.2f}%",
            "CAGR": f"{m.cagr * 100:.2f}%",
            "Annualized Volatility": f"{m.annualized_volatility * 100:.2f}%",
            "Sharpe Ratio": f"{m.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{m.sortino_ratio:.2f}",
            "Max Drawdown": f"{m.max_drawdown * 100:.2f}%",
            "Max DD Duration": f"{m.max_drawdown_duration_bars} bars",
            "Total Trades": str(m.total_trades),
            "Win Rate": f"{m.win_rate * 100:.1f}%",
            "Profit Factor": f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "inf",
            "Total Fees (INR)": f"{m.total_fees:,.2f}",
            "Total Slippage (INR)": f"{m.total_slippage_cost:,.2f}",
            "Turnover": f"{m.turnover:.2f}x",
        }
    return pd.DataFrame(data)


def export_result_json(result: SimulationResult, filepath: Optional[str] = None) -> str:
    """Serialize SimulationResult metrics and trade log to JSON.
    
    Args:
        result: SimulationResult instance.
        filepath: Optional output file path.
        
    Returns:
        JSON string representation.
    """
    payload = {
        "strategy_name": result.strategy_name,
        "config": {
            "initial_cash": result.config.initial_cash,
            "fee_rate": result.config.fee_rate,
            "slippage_rate": result.config.slippage_rate,
            "min_trade_size": result.config.min_trade_size,
            "allow_fractional": result.config.allow_fractional,
            "risk_free_rate": result.config.risk_free_rate,
        },
        "metrics": result.metrics.to_dict(),
        "trade_count": len(result.trades_df),
        "order_count": len(result.orders_df),
        "bar_count": len(result.snapshots_df),
    }
    dumped = json.dumps(payload, indent=2)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(dumped)
    return dumped
