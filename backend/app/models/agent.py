"""Agent Run Model for TradeMind (Phase 9).

Stores PPO training and evaluation metadata, referencing model artifact paths
without storing large model binaries inside SQLite.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..db import db, safe_json_loads, safe_serialize


class AgentRun(db.Model):
    """Execution record for PPO training and offline policy evaluation."""
    __tablename__ = "agent_runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    symbol = db.Column(db.String(32), index=True, nullable=False)
    status = db.Column(db.String(32), default="completed", index=True, nullable=False)
    model_path = db.Column(db.String(256), nullable=True)
    configuration_json = db.Column(db.Text, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent run to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "symbol": self.symbol,
            "status": self.status,
            "model_path": self.model_path,
            "configuration": safe_json_loads(self.configuration_json, {}),
            "metrics": safe_json_loads(self.metrics_json, None),
            "created_at": safe_serialize(self.created_at),
            "completed_at": safe_serialize(self.completed_at),
            "error_message": self.error_message,
        }
