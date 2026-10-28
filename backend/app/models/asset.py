"""Asset Model for TradeMind (Phase 9).

Stores canonical reference details for supported trading assets.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from ..db import db, safe_serialize


class Asset(db.Model):
    """Supported trading asset entity."""
    __tablename__ = "assets"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String(32), unique=True, index=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    exchange = db.Column(db.String(32), default="NSE", nullable=False)
    asset_type = db.Column(db.String(32), default="EQUITY", nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("symbol", "exchange", name="uq_asset_symbol_exchange"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert asset model to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "asset_type": self.asset_type,
            "created_at": safe_serialize(self.created_at),
        }
