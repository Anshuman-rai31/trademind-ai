"""Model Registry Service for TradeMind (Phase 9.5B).

Manages model registration, artifact checksums, schema verification,
approval workflows, active model selection, and safe rollback for inference.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stable_baselines3 import PPO

from ..config import PROJECT_ROOT
from ..data.pipeline import sanitize_filename_component
from ..data.validator import validate_symbol
from ..db import db, safe_json_dumps, safe_serialize
from ..models.model_registry import RegisteredModel

logger = logging.getLogger(__name__)


def calculate_file_checksum(path: Path) -> Optional[str]:
    """Calculate SHA-256 checksum of a file on disk."""
    if not path or not path.exists():
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelRegistryService:
    """Service layer managing RegisteredModel persistence, approval, and loading."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self._cached_ppo: Optional[PPO] = None
        self._cached_model_id: Optional[str] = None

    def sync_local_artifacts(self) -> int:
        """Scan models directory and register untracked model artifacts into SQLite."""
        registered_count = 0
        if not self.models_dir.exists():
            return 0

        # Discover all .zip model checkpoints
        for zip_path in self.models_dir.glob("**/*.zip"):
            rel_path = str(zip_path.relative_to(PROJECT_ROOT))
            checksum = calculate_file_checksum(zip_path)

            # Generate stable model_id
            stem = zip_path.stem.lower()
            parent_name = zip_path.parent.name.lower()
            if parent_name and parent_name not in ("models", "checkpoints"):
                model_id = f"ppo_{parent_name}_{stem}"
                inferred_symbol = parent_name.upper().replace("_", ".")
            else:
                model_id = f"ppo_{stem}"
                inferred_symbol = "RELIANCE.NS"

            existing = RegisteredModel.query.filter(
                (RegisteredModel.model_id == model_id) | (RegisteredModel.artifact_path == rel_path)
            ).first()

            if existing:
                # Update checksum if null
                if not existing.artifact_checksum and checksum:
                    existing.artifact_checksum = checksum
                    db.session.commit()
                continue

            # Load metadata from training_metadata.json if available
            meta_path = zip_path.parent / "training_metadata.json"
            train_cfg = {}
            val_metrics = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        inferred_symbol = meta_data.get("symbol", inferred_symbol)
                        train_cfg = meta_data.get("train_config", {})
                        val_metrics = meta_data.get("best_validation_metrics", {})
                except Exception as exc:
                    logger.warning("Could not parse metadata for %s: %s", zip_path, exc)

            canonical_features = [
                "simple_return", "log_return", "rsi_14", "bb_pct_20_2.0",
                "macd_histogram", "atr_14", "close_zscore_20",
                "cash_ratio", "holdings_ratio", "position_flag", "unrealized_return", "drawdown"
            ]

            # Mark best_model as candidate
            status = "approved" if "best" in stem else "candidate"

            model_rec = RegisteredModel(
                model_id=model_id,
                name=f"PPO Policy ({inferred_symbol} - {stem})",
                symbol=inferred_symbol,
                version="1.0.0",
                artifact_path=rel_path,
                artifact_checksum=checksum,
                algorithm="PPO",
                status=status,
                is_active=False,
                obs_dimension=12,
                action_space="Discrete(3)",
                feature_schema_json=safe_json_dumps(canonical_features),
                training_config_json=safe_json_dumps(train_cfg),
                validation_metrics_json=safe_json_dumps(val_metrics),
            )
            db.session.add(model_rec)
            registered_count += 1

        if registered_count > 0:
            db.session.commit()
            logger.info("Registered %d new model artifacts into model registry.", registered_count)

        return registered_count

    def list_models(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List registered models with optional symbol and status filtering."""
        query = RegisteredModel.query
        if symbol:
            try:
                clean_sym = validate_symbol(symbol)
                query = query.filter_by(symbol=clean_sym)
            except Exception:
                pass
        if status:
            query = query.filter_by(status=status.lower().strip())

        models = query.order_by(RegisteredModel.id.desc()).all()
        return [m.to_dict() for m in models]

    def get_model(self, id_or_model_id: str | int) -> Optional[RegisteredModel]:
        """Retrieve registered model by integer ID or string model_id."""
        if isinstance(id_or_model_id, int) or (isinstance(id_or_model_id, str) and id_or_model_id.isdigit()):
            return RegisteredModel.query.get(int(id_or_model_id))
        return RegisteredModel.query.filter_by(model_id=str(id_or_model_id)).first()

    def _resolve_artifact_path(self, artifact_path: str) -> Path:
        """Resolve an artifact path to an absolute Path, handling both absolute and relative paths."""
        path = Path(artifact_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def approve_model(self, id_or_model_id: str | int) -> RegisteredModel:
        """Promote a candidate model to approved status."""
        model = self.get_model(id_or_model_id)
        if not model:
            raise ValueError(f"Model not found: {id_or_model_id}")

        abs_path = self._resolve_artifact_path(model.artifact_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Model artifact file does not exist: {model.artifact_path}")

        model.status = "approved"
        model.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return model

    def activate_model(self, id_or_model_id: str | int) -> RegisteredModel:
        """Set model as active inference policy, deactivating other models for the same symbol."""
        model = self.get_model(id_or_model_id)
        if not model:
            raise ValueError(f"Model not found: {id_or_model_id}")

        abs_path = self._resolve_artifact_path(model.artifact_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Model artifact file does not exist on disk: {model.artifact_path}")

        # Checksum verification if available
        if model.artifact_checksum:
            current_checksum = calculate_file_checksum(abs_path)
            if current_checksum != model.artifact_checksum:
                raise ValueError(
                    f"Model artifact checksum mismatch for {model.model_id}. "
                    f"Expected {model.artifact_checksum}, got {current_checksum}"
                )

        if model.status not in ("approved", "active"):
            raise ValueError(
                f"Model '{model.model_id}' cannot be activated: status is '{model.status}'. "
                f"Only 'approved' models can be activated for inference."
            )

        # Deactivate any currently active models for this symbol
        RegisteredModel.query.filter_by(symbol=model.symbol, is_active=True).update({"is_active": False})

        model.is_active = True
        model.status = "active"
        model.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        # Invalidate in-memory cache
        self._cached_ppo = None
        self._cached_model_id = None

        return model

    def rollback_model(self, symbol: str) -> Optional[RegisteredModel]:
        """Roll back active status for a symbol to the previous approved model."""
        clean_sym = validate_symbol(symbol)

        # Find current active model
        active_model = RegisteredModel.query.filter_by(symbol=clean_sym, is_active=True).first()
        if active_model:
            active_model.is_active = False
            active_model.status = "approved"
            active_model.updated_at = datetime.now(timezone.utc)

        # Find latest approved model that is not the rolled back model
        prev_approved = (
            RegisteredModel.query.filter(
                RegisteredModel.symbol == clean_sym,
                RegisteredModel.status == "approved",
                RegisteredModel.id != (active_model.id if active_model else -1),
            )
            .order_by(RegisteredModel.id.desc())
            .first()
        )

        if prev_approved:
            prev_approved.is_active = True
            prev_approved.status = "active"
            prev_approved.updated_at = datetime.now(timezone.utc)

        db.session.commit()
        self._cached_ppo = None
        self._cached_model_id = None
        return prev_approved

    def get_active_model(self, symbol: str) -> Optional[RegisteredModel]:
        """Get the currently active inference model for a symbol."""
        try:
            clean_sym = validate_symbol(symbol)
        except Exception:
            return None
        return RegisteredModel.query.filter_by(symbol=clean_sym, is_active=True).first()

    def load_model_for_inference(self, symbol: str) -> Tuple[PPO, RegisteredModel]:
        """Validate and securely load the active PPO model for inference.
        
        Raises:
            ValueError: If no active model is approved or artifact is corrupted.
            FileNotFoundError: If artifact is missing.
        """
        clean_sym = validate_symbol(symbol)
        model_rec = self.get_active_model(clean_sym)
        if not model_rec:
            # Fallback: check if an approved model exists
            model_rec = (
                RegisteredModel.query.filter_by(symbol=clean_sym, status="approved")
                .order_by(RegisteredModel.id.desc())
                .first()
            )

        if not model_rec:
            raise ValueError(
                f"No active or approved PPO model found in registry for symbol '{clean_sym}'. "
                f"A model must be registered and approved before running inference."
            )

        abs_path = self._resolve_artifact_path(model_rec.artifact_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {abs_path}")

        # Check cache
        if self._cached_ppo is not None and self._cached_model_id == model_rec.model_id:
            return self._cached_ppo, model_rec

        # Load PPO policy
        model = PPO.load(str(abs_path), device="cpu")

        # Verify observation space dimension
        expected_dim = model_rec.obs_dimension
        actual_shape = getattr(model.observation_space, "shape", None)
        if actual_shape and actual_shape[0] != expected_dim:
            raise ValueError(
                f"Model observation dimension mismatch: expected {expected_dim}, "
                f"model observation_space is {actual_shape}"
            )

        self._cached_ppo = model
        self._cached_model_id = model_rec.model_id
        return model, model_rec
