"""RL Trading Signal Database Model (Phase 12).

Persists model-generated BUY, SELL, and HOLD trading signals with full auditability:
raw model proposed action vs. risk-effective action, exact 12-d observation vectors,
source candle timestamps, model artifact checksums, and rejection/risk reasons.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint

from ..db import db, safe_json_dumps, safe_json_loads, safe_serialize


class RLTradingSignal(db.Model):
    """Database model for real-time RL model-generated trading signals."""

    __tablename__ = "rl_trading_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(40), unique=True, nullable=False, index=True, default=lambda: f"sig_{uuid.uuid4().hex[:16]}")
    symbol = Column(String(32), nullable=False, index=True)
    portfolio_id = Column(Integer, ForeignKey("paper_portfolios.id"), nullable=True, index=True)
    interval = Column(String(8), nullable=False, default="1d", index=True)
    
    # Model decision and Risk intervention results
    action = Column(String(16), nullable=False, default="HOLD")               # Primary displayed action
    proposed_action = Column(String(16), nullable=True)                      # Raw model prediction before risk check
    effective_action = Column(String(16), nullable=False, default="HOLD")    # Action after risk manager evaluation
    confidence = Column(Float, nullable=True)                                # None for standard PPO (no fabricated values)
    close_price = Column(Float, nullable=False)                              # Reference price from completed candle
    
    # Candle and timing lineage
    source_candle_timestamp = Column(String(32), nullable=False, index=True) # ISO-8601 timestamp of completed candle
    source_candle_start = Column(String(32), nullable=True)                  # Candle window start
    source_candle_end = Column(String(32), nullable=True)                    # Candle window end
    timestamp = Column(String(32), nullable=False, index=True)               # Signal generation timestamp (ISO-8601 UTC)
    
    # Model provenance
    model_id = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=True)
    model_checksum = Column(String(64), nullable=True)
    
    # Telemetry and validity status
    data_freshness = Column(String(16), nullable=True, default="FRESH")      # FRESH, AGING, STALE
    status = Column(String(32), nullable=False, default="VALID", index=True) # VALID, REJECTED, RISK_BLOCKED, RISK_MODIFIED, TIMEFRAME_MISMATCH
    rejection_reason = Column(Text, nullable=True)
    
    # Audit payloads
    observation_json = Column(Text, nullable=True)                           # 12-d observation vector
    features_json = Column(Text, nullable=True)                              # Named breakdown of technical & portfolio features
    risk_evaluation_json = Column(Text, nullable=True)                       # Pre-trade risk manager details
    is_mock = Column(Boolean, nullable=False, default=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "source_candle_timestamp", name="uq_signal_symbol_interval_candle"),
        Index("idx_signal_lookup", "symbol", "interval", "timestamp"),
        Index("idx_signal_status_time", "status", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal entity to serialized dictionary."""
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "portfolio_id": self.portfolio_id,
            "interval": self.interval,
            "action": self.action,
            "proposed_action": self.proposed_action,
            "effective_action": self.effective_action,
            "confidence": self.confidence,
            "close_price": round(float(self.close_price), 2) if self.close_price is not None else None,
            "source_candle_timestamp": self.source_candle_timestamp,
            "source_candle_start": self.source_candle_start,
            "source_candle_end": self.source_candle_end,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_checksum": self.model_checksum,
            "data_freshness": self.data_freshness,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "observation": safe_json_loads(self.observation_json, None),
            "features": safe_json_loads(self.features_json, None),
            "risk_evaluation": safe_json_loads(self.risk_evaluation_json, None),
            "is_mock": self.is_mock,
            "created_at": safe_serialize(self.created_at),
        }
