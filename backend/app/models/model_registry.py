"""Model Registry ORM Model for TradeMind (Phase 9.5B).

Tracks trained RL and ML model artifacts, versioning, feature schemas, checksums,
and operational lifecycle states (candidate, approved, active, archived, failed).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db import db, safe_json_loads, safe_serialize


class RegisteredModel(db.Model):
    """Registered model entity tracking ML/RL model artifacts and operational status."""
    __tablename__ = "registered_models"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_id = db.Column(db.String(64), unique=True, index=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    symbol = db.Column(db.String(32), index=True, nullable=False)
    version = db.Column(db.String(32), default="1.0.0", nullable=False)
    artifact_path = db.Column(db.String(256), nullable=False)
    artifact_checksum = db.Column(db.String(64), nullable=True)  # SHA-256
    algorithm = db.Column(db.String(32), default="PPO", nullable=False)
    status = db.Column(
        db.String(32),
        default="candidate",
        index=True,
        nullable=False,
    )  # candidate, approved, active, archived, failed
    is_active = db.Column(db.Boolean, default=False, index=True, nullable=False)
    obs_dimension = db.Column(db.Integer, default=12, nullable=False)
    action_space = db.Column(db.String(64), default="Discrete(3)", nullable=False)
    feature_schema_json = db.Column(db.Text, nullable=True)
    training_config_json = db.Column(db.Text, nullable=True)
    validation_metrics_json = db.Column(db.Text, nullable=True)
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert registered model to dictionary representation."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "name": self.name,
            "symbol": self.symbol,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "artifact_checksum": self.artifact_checksum,
            "algorithm": self.algorithm,
            "status": self.status,
            "is_active": self.is_active,
            "obs_dimension": self.obs_dimension,
            "action_space": self.action_space,
            "feature_schema": safe_json_loads(self.feature_schema_json, []),
            "training_config": safe_json_loads(self.training_config_json, {}),
            "validation_metrics": safe_json_loads(self.validation_metrics_json, {}),
            "created_at": safe_serialize(self.created_at),
            "updated_at": safe_serialize(self.updated_at),
        }
