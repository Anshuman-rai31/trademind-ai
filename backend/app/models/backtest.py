"""Backtest, Trade, and Portfolio Snapshot Models for TradeMind (Phase 9).

Persists backtest executions, individual simulated order fills (trades),
and daily mark-to-market portfolio equity snapshots.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db import db, safe_json_loads, safe_serialize


class BacktestRun(db.Model):
    """Metadata and execution results for a quantitative backtest."""
    __tablename__ = "backtest_runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    symbol = db.Column(db.String(32), index=True, nullable=False)
    strategy_type = db.Column(db.String(64), index=True, nullable=False)
    start_date = db.Column(db.String(32), nullable=False)
    end_date = db.Column(db.String(32), nullable=False)
    initial_capital = db.Column(db.Float, default=100000.0, nullable=False)
    status = db.Column(db.String(32), default="queued", index=True, nullable=False)
    configuration_json = db.Column(db.Text, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    trades = db.relationship(
        "TradeRecord",
        backref="backtest",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="TradeRecord.id",
    )
    portfolio_snapshots = db.relationship(
        "PortfolioSnapshotRecord",
        backref="backtest",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="PortfolioSnapshotRecord.id",
    )

    def to_dict(self, include_metrics: bool = True) -> Dict[str, Any]:
        """Convert backtest run to dictionary."""
        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "status": self.status,
            "configuration": safe_json_loads(self.configuration_json, {}),
            "created_at": safe_serialize(self.created_at),
            "completed_at": safe_serialize(self.completed_at),
            "error_message": self.error_message,
        }
        if include_metrics:
            data["metrics"] = safe_json_loads(self.metrics_json, None)
        return data


class TradeRecord(db.Model):
    """Persisted trade execution record generated during a backtest."""
    __tablename__ = "trades"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    backtest_id = db.Column(
        db.Integer,
        db.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    symbol = db.Column(db.String(32), index=True, nullable=False)
    timestamp = db.Column(db.String(64), nullable=False)
    action = db.Column(db.String(16), nullable=False)  # BUY or SELL
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    transaction_cost = db.Column(db.Float, default=0.0, nullable=False)
    realized_pnl = db.Column(db.Float, nullable=True)
    metadata_json = db.Column(db.Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary."""
        return {
            "id": self.id,
            "backtest_id": self.backtest_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "transaction_cost": self.transaction_cost,
            "realized_pnl": self.realized_pnl,
            "metadata": safe_json_loads(self.metadata_json, {}),
        }


class PortfolioSnapshotRecord(db.Model):
    """Persisted daily portfolio valuation mark-to-market snapshot."""
    __tablename__ = "portfolio_snapshots"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    backtest_id = db.Column(
        db.Integer,
        db.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    timestamp = db.Column(db.String(64), nullable=False)
    cash = db.Column(db.Float, nullable=False)
    holdings_value = db.Column(db.Float, default=0.0, nullable=False)
    portfolio_value = db.Column(db.Float, nullable=False)
    drawdown = db.Column(db.Float, default=0.0, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio snapshot to dictionary."""
        return {
            "id": self.id,
            "backtest_id": self.backtest_id,
            "timestamp": self.timestamp,
            "cash": self.cash,
            "holdings_value": self.holdings_value,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.drawdown,
        }
