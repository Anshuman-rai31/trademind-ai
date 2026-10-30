"""Paper Trading and Simulation Persistence Models for TradeMind (Phase 9.5B).

Persists paper trading portfolios, open positions, executed orders, and pre-trade
risk intervention event audit trails in SQLite.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db import db, safe_json_loads, safe_serialize


class PaperPortfolio(db.Model):
    """Simulated or paper-trading portfolio entity."""
    __tablename__ = "paper_portfolios"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), unique=True, index=True, default="default", nullable=False)
    initial_cash = db.Column(db.Float, default=100000.0, nullable=False)
    cash = db.Column(db.Float, default=100000.0, nullable=False)
    portfolio_value = db.Column(db.Float, default=100000.0, nullable=False)
    peak_value = db.Column(db.Float, default=100000.0, nullable=False)
    total_realized_pnl = db.Column(db.Float, default=0.0, nullable=False)
    total_trades_count = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    allocation_method = db.Column(db.String(32), default="EQUAL_WEIGHT", nullable=False)  # EQUAL_WEIGHT, CUSTOM
    strategy_type = db.Column(db.String(64), default="BUY_AND_HOLD", nullable=False)  # BUY_AND_HOLD, PPO_AGENT, CUSTOM_WEIGHTS, MA_CROSSOVER
    risk_config_json = db.Column(db.Text, nullable=True)
    description = db.Column(db.String(256), nullable=True)
    user_id = db.Column(db.String(64), default="default_user", nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)  # Automated paper-trading toggle

    bucket_items = db.relationship(
        "PortfolioBucketItem",
        backref="portfolio",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="PortfolioBucketItem.id.asc()",
    )
    snapshots = db.relationship(
        "PaperSnapshotRecord",
        backref="portfolio",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="PaperSnapshotRecord.id.asc()",
    )
    positions = db.relationship(
        "PaperPosition",
        backref="portfolio",
        cascade="all, delete-orphan",
        lazy="select",
    )
    trades = db.relationship(
        "PaperTrade",
        backref="portfolio",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="PaperTrade.id.desc()",
    )
    risk_events = db.relationship(
        "RiskEventRecord",
        backref="portfolio",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="RiskEventRecord.id.desc()",
    )

    def to_dict(self, include_positions: bool = True, include_bucket: bool = True) -> Dict[str, Any]:
        """Convert portfolio to dictionary with full metrics and bucket details."""
        holdings_val = sum((p.quantity * p.current_price) for p in self.positions if p.quantity > 0)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions if p.quantity > 0)
        total_pnl = self.total_realized_pnl + unrealized_pnl
        return_pct = ((self.portfolio_value - self.initial_cash) / self.initial_cash * 100.0) if self.initial_cash > 0 else 0.0
        drawdown_pct = max(0.0, ((self.peak_value - self.portfolio_value) / self.peak_value * 100.0)) if self.peak_value > 0 else 0.0

        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "description": self.description,
            "allocation_method": self.allocation_method,
            "strategy_type": self.strategy_type,
            "risk_config": safe_json_loads(self.risk_config_json, {}),
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "holdings_value": round(holdings_val, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "peak_value": round(self.peak_value, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "drawdown_pct": round(drawdown_pct, 2),
            "total_trades_count": self.total_trades_count,
            "is_active": getattr(self, "is_active", True),
            "created_at": safe_serialize(self.created_at),
            "updated_at": safe_serialize(self.updated_at),
        }
        if include_positions:
            data["positions"] = [p.to_dict() for p in self.positions if p.quantity > 0]
        if include_bucket:
            data["bucket"] = [b.to_dict() for b in self.bucket_items]
        return data


class PortfolioBucketItem(db.Model):
    """Target asset allocation item within a user-configured portfolio bucket/cart."""
    __tablename__ = "portfolio_bucket_items"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(
        db.Integer,
        db.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    symbol = db.Column(db.String(32), index=True, nullable=False)
    target_weight = db.Column(db.Float, default=0.0, nullable=False)  # Normalized fraction (0.0 to 1.0)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("portfolio_id", "symbol", name="uq_portfolio_bucket_symbol"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert bucket item to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "target_weight": round(self.target_weight, 4),
            "target_pct": round(self.target_weight * 100.0, 2),
            "created_at": safe_serialize(self.created_at),
            "updated_at": safe_serialize(self.updated_at),
        }


class PaperSnapshotRecord(db.Model):
    """Persisted portfolio valuation snapshot for equity curve visualization."""
    __tablename__ = "paper_portfolio_snapshots"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(
        db.Integer,
        db.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    timestamp = db.Column(db.String(64), nullable=False)
    cash = db.Column(db.Float, nullable=False)
    holdings_value = db.Column(db.Float, default=0.0, nullable=False)
    portfolio_value = db.Column(db.Float, nullable=False)
    drawdown = db.Column(db.Float, default=0.0, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot record to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp,
            "cash": round(self.cash, 2),
            "holdings_value": round(self.holdings_value, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "drawdown": round(self.drawdown, 4),
            "created_at": safe_serialize(self.created_at),
        }


class PaperPosition(db.Model):
    """Open asset position held within a simulation portfolio."""
    __tablename__ = "paper_positions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(
        db.Integer,
        db.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    symbol = db.Column(db.String(32), index=True, nullable=False)
    quantity = db.Column(db.Float, default=0.0, nullable=False)
    average_entry_price = db.Column(db.Float, default=0.0, nullable=False)
    current_price = db.Column(db.Float, default=0.0, nullable=False)
    unrealized_pnl = db.Column(db.Float, default=0.0, nullable=False)
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("portfolio_id", "symbol", name="uq_portfolio_symbol_position"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_entry_price": round(self.average_entry_price, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.quantity * self.current_price, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "updated_at": safe_serialize(self.updated_at),
        }


class PaperTrade(db.Model):
    """Executed simulated trade record."""
    __tablename__ = "paper_trades"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(
        db.Integer,
        db.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    signal_id = db.Column(db.String(40), index=True, nullable=True)  # Lineage to RLTradingSignal
    symbol = db.Column(db.String(32), index=True, nullable=False)
    timestamp = db.Column(db.String(64), nullable=False)
    action = db.Column(db.String(16), nullable=False)  # BUY, SELL, HOLD
    quantity = db.Column(db.Float, default=0.0, nullable=False)
    price = db.Column(db.Float, default=0.0, nullable=False)
    fee = db.Column(db.Float, default=0.0, nullable=False)
    slippage = db.Column(db.Float, default=0.0, nullable=False)
    realized_pnl = db.Column(db.Float, default=0.0, nullable=False)
    risk_status = db.Column(db.String(32), default="APPROVED", nullable=False)  # APPROVED, MODIFIED, BLOCKED
    metadata_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "action": self.action,
            "quantity": self.quantity,
            "price": round(self.price, 2),
            "fee": round(self.fee, 2),
            "slippage": round(self.slippage, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "risk_status": self.risk_status,
            "metadata": safe_json_loads(self.metadata_json, {}),
            "created_at": safe_serialize(self.created_at),
        }


class RiskEventRecord(db.Model):
    """Persisted record of pre-trade risk constraint evaluations and interventions."""
    __tablename__ = "risk_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    portfolio_id = db.Column(
        db.Integer,
        db.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    symbol = db.Column(db.String(32), index=True, nullable=False)
    timestamp = db.Column(db.String(64), nullable=False)
    rule_breached = db.Column(db.String(64), nullable=False)
    severity = db.Column(db.String(32), default="INTERVENTION", nullable=False)  # WARNING, INTERVENTION, BLOCKED
    proposed_action = db.Column(db.String(32), nullable=False)
    effective_action = db.Column(db.String(32), nullable=False)
    details = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert risk event to dictionary."""
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "rule_breached": self.rule_breached,
            "severity": self.severity,
            "proposed_action": self.proposed_action,
            "effective_action": self.effective_action,
            "details": self.details,
            "created_at": safe_serialize(self.created_at),
        }
