"""Market-Regime Detection Service for TradeMind (Phase 9).

Integrates the Phase 8 MarketRegimeDetector into API endpoints, loading fitted
model artifacts from disk and returning historical regime classifications and cluster summaries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import PROJECT_ROOT
from ..data.pipeline import sanitize_filename_component
from ..data.validator import validate_symbol
from ..regime.detector import MarketRegimeDetector
from .market_data_service import MarketDataService

logger = logging.getLogger(__name__)


class RegimeService:
    """Service layer interfacing with MarketRegimeDetector."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        data_service: Optional[MarketDataService] = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self.data_service = data_service or MarketDataService()

    def find_model_path(self, symbol: str) -> Optional[Path]:
        """Locate fitted regime detector artifact on disk for a given symbol."""
        try:
            clean_sym = validate_symbol(symbol)
            safe_sym = sanitize_filename_component(clean_sym)
        except Exception:
            return None

        # 1. Symbol-specific model: regime_detector_reliance_ns.joblib
        specific_path = self.models_dir / f"regime_detector_{safe_sym}.joblib"
        if specific_path.exists():
            return specific_path

        # 2. General fitted model: regime_detector.joblib
        general_path = self.models_dir / "regime_detector.joblib"
        if general_path.exists():
            return general_path

        return None

    def get_regime_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return model metadata and configuration for the symbol's regime detector."""
        clean_sym = validate_symbol(symbol)
        model_path = self.find_model_path(clean_sym)
        if not model_path:
            return None

        try:
            detector = MarketRegimeDetector.load(str(model_path))
            meta = detector.metadata.to_dict() if detector.metadata else {}
            return {
                "symbol": clean_sym,
                "model_available": True,
                "model_path": model_path.name,
                "metadata": meta,
                "n_clusters": detector.config.n_clusters,
                "feature_names": list(detector.config.feature_names),
            }
        except Exception as exc:
            logger.warning("Error loading regime detector from %s: %s", model_path, exc)
            return None

    def get_cluster_summaries(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Return empirical cluster summaries and descriptive labels for fitted detector."""
        clean_sym = validate_symbol(symbol)
        model_path = self.find_model_path(clean_sym)
        if not model_path:
            return None

        try:
            detector = MarketRegimeDetector.load(str(model_path))
            return [s.to_dict() for s in detector.cluster_summaries]
        except Exception as exc:
            logger.warning("Error loading cluster summaries from %s: %s", model_path, exc)
            return None

    def get_latest_regime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compute regime classification for the latest available historical bar.
        
        Guarantees:
            - Never retrains detector automatically during API calls.
            - Explicitly notes that regime classification is retrospective, not predictive.
        """
        clean_sym = validate_symbol(symbol)
        model_path = self.find_model_path(clean_sym)
        if not model_path:
            return None

        df = self.data_service.get_historical_dataframe(clean_sym)
        if df is None or len(df) < 5:
            return None

        try:
            detector = MarketRegimeDetector.load(str(model_path))
            labeled_df = detector.predict_df(df)
            last_row = labeled_df.iloc[-1]

            return {
                "symbol": clean_sym,
                "timestamp": str(last_row["timestamp"]),
                "regime_cluster": int(last_row["regime_cluster"]),
                "regime_label": str(last_row["regime_label"]),
                "model_path": model_path.name,
                "disclaimer": (
                    "Market regime classifications are retrospective historical patterns "
                    "derived from unsupervised clustering. They do not predict future returns."
                ),
            }
        except Exception as exc:
            logger.warning("Error classifying latest regime for %s: %s", clean_sym, exc)
            return None
