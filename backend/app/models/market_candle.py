"""Market Candle Database Model (Phase 11D).

Persists completed OHLCV candles constructed from real-time streaming feeds,
with interval indexing and uniqueness constraints preventing duplicates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, UniqueConstraint

from ..db import db


class MarketCandle(db.Model):
    """Database model for completed intraday and daily OHLCV market candles."""

    __tablename__ = "market_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    interval = Column(String(8), nullable=False, default="5m", index=True)  # "1m", "5m", "1d"
    start_time = Column(String(32), nullable=False, index=True)  # ISO-8601 string
    end_time = Column(String(32), nullable=False)                # ISO-8601 string
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False, default=0)
    source = Column(String(32), nullable=False, default="kite")
    is_complete = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "start_time", name="uq_candle_symbol_interval_start"),
        Index("idx_candle_lookup", "symbol", "interval", "start_time"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert candle record to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "interval": self.interval,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": self.volume,
            "source": self.source,
            "is_complete": self.is_complete,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
