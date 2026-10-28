"""PPO Agent Service for TradeMind (Phase 9).

Integrates the Phase 6 Reinforcement Learning PPO Agent, model artifact inspection,
offline evaluation metrics, and deterministic inference decisions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from stable_baselines3 import PPO

from ..config import PROJECT_ROOT
from ..data.pipeline import sanitize_filename_component
from ..data.validator import validate_symbol
from ..models.agent import AgentRun
from ..rl.actions import TradingAction

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """Raised when an inference decision is requested but no loadable model artifact exists."""
    pass


class AgentService:
    """Service layer managing PPO model loading, status, and inference."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self._cached_model: Optional[PPO] = None
        self._cached_model_path: Optional[str] = None

    def find_model_artifact(self, symbol: Optional[str] = None) -> Optional[Path]:
        """Locate a trained PPO model artifact (.zip) in models directory."""
        if symbol:
            try:
                clean_sym = validate_symbol(symbol)
                safe_sym = sanitize_filename_component(clean_sym)
                # Check symbol-specific artifacts
                candidates = [
                    self.models_dir / f"ppo_{safe_sym}.zip",
                    self.models_dir / safe_sym / "best_model.zip",
                    self.models_dir / safe_sym / "final_model.zip",
                ]
                for p in candidates:
                    if p.exists():
                        return p
            except Exception:
                pass

        # Check default top-level model files
        default_candidates = [
            self.models_dir / "best_model.zip",
            self.models_dir / "final_model.zip",
        ]
        for p in default_candidates:
            if p.exists():
                return p

        # Check any .zip file inside models directory
        all_zips = sorted(self.models_dir.glob("**/*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if all_zips:
            return all_zips[0]

        return None

    def get_agent_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Distinguish between installation, artifact availability, and inference readiness."""
        artifact_path = self.find_model_artifact(symbol)
        artifact_exists = artifact_path is not None and artifact_path.exists()
        is_loadable = False
        meta_data: Optional[Dict[str, Any]] = None

        if artifact_exists and artifact_path:
            # Check companion metadata
            meta_file = artifact_path.parent / "training_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                except Exception:
                    pass

            # Test if model is loadable
            try:
                _ = self._load_model(artifact_path)
                is_loadable = True
            except Exception as exc:
                logger.warning("Model artifact exists at %s but failed to load: %s", artifact_path, exc)
                is_loadable = False

        latest_run = AgentRun.query.order_by(AgentRun.id.desc()).first()

        return {
            "training_support_installed": True,
            "algorithm": "PPO",
            "framework": "Stable-Baselines3",
            "saved_model_artifact_available": artifact_exists,
            "ready_for_inference": is_loadable,
            "model_artifact_path": str(artifact_path.name) if artifact_path else None,
            "metadata": meta_data,
            "latest_agent_run": latest_run.to_dict() if latest_run else None,
        }

    def get_persisted_metrics(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return actual persisted evaluation metrics from database or training metadata."""
        # 1. Query latest AgentRun record
        query = AgentRun.query
        if symbol:
            try:
                clean_sym = validate_symbol(symbol)
                query = query.filter_by(symbol=clean_sym)
            except Exception:
                pass

        latest_run = query.order_by(AgentRun.id.desc()).first()
        if latest_run and latest_run.metrics:
            return {
                "source": "database",
                "run_id": latest_run.id,
                "name": latest_run.name,
                "symbol": latest_run.symbol,
                "metrics": latest_run.metrics,
                "completed_at": latest_run.completed_at.isoformat() if latest_run.completed_at else None,
            }

        # 2. Check metadata file in models directory
        artifact = self.find_model_artifact(symbol)
        if artifact:
            meta_file = artifact.parent / "training_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "best_validation_metrics" in data and data["best_validation_metrics"]:
                            return {
                                "source": "artifact_metadata",
                                "symbol": data.get("symbol", symbol or "UNKNOWN"),
                                "metrics": data["best_validation_metrics"],
                                "file": meta_file.name,
                            }
                except Exception:
                    pass

        return None

    def predict_action(
        self,
        symbol: str,
        state: Union[List[float], Dict[str, float]],
    ) -> Dict[str, Any]:
        """Generate a trading action from the PPO model without automatic training."""
        clean_sym = validate_symbol(symbol)
        artifact_path = self.find_model_artifact(clean_sym)
        if not artifact_path or not artifact_path.exists():
            raise ModelUnavailableError(
                f"No trained PPO model artifact found for symbol '{clean_sym}'. "
                f"A trained model artifact (.zip) must be placed in models/ before requesting decisions."
            )

        model = self._load_model(artifact_path)

        # Parse and format input state vector
        if isinstance(state, dict):
            # Extract values in stable sorted key order
            obs_values = [float(v) for _, v in sorted(state.items())]
        elif isinstance(state, (list, tuple)):
            obs_values = [float(v) for v in state]
        else:
            raise ValueError(f"State must be a list of floats or a dictionary of features, got {type(state)}")

        obs_array = np.array(obs_values, dtype=np.float32)

        # Verify observation shape matches model expectation if possible
        expected_shape = getattr(model.observation_space, "shape", None)
        if expected_shape and len(expected_shape) > 0:
            if len(obs_array) != expected_shape[0]:
                raise ValueError(
                    f"Observation dimension mismatch: model expects {expected_shape[0]} features, "
                    f"received {len(obs_array)}."
                )

        # Run deterministic model prediction
        action_idx, _ = model.predict(obs_array, deterministic=True)
        int_action = int(action_idx)
        try:
            action_name = TradingAction(int_action).name
        except ValueError:
            action_name = f"ACTION_{int_action}"

        return {
            "symbol": clean_sym,
            "action": int_action,
            "action_name": action_name,
            "model_artifact": artifact_path.name,
            "disclaimer": (
                "This decision is an experimental model output generated for quantitative research "
                "and simulation purposes only. It does not constitute financial advice or a guarantee "
                "of future price performance."
            ),
        }

    def _load_model(self, path: Path) -> PPO:
        """Load and cache PPO model instance from disk."""
        path_str = str(path.resolve())
        if self._cached_model is not None and self._cached_model_path == path_str:
            return self._cached_model

        model = PPO.load(path_str, device="cpu")
        self._cached_model = model
        self._cached_model_path = path_str
        return model
