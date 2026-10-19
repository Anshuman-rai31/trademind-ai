"""TradeMind Phase 8 Market-Regime Detection Package.

Provides unsupervised, leakage-safe market regime classification and profiling
for quantitative time-series research and simulation.
"""
from __future__ import annotations

from .config import RegimeConfig
from .detector import MarketRegimeDetector, compute_scale_free_regime_features
from .exceptions import (
    DataLeakageError,
    FeatureExtractionError,
    InsufficientDataError,
    ModelNotFittedError,
    RegimeDetectionError,
)
from .models import ClusterSummary, RegimeModelMetadata, RegimeOutput

__all__ = [
    "RegimeConfig",
    "MarketRegimeDetector",
    "compute_scale_free_regime_features",
    "ClusterSummary",
    "RegimeModelMetadata",
    "RegimeOutput",
    "RegimeDetectionError",
    "ModelNotFittedError",
    "InsufficientDataError",
    "FeatureExtractionError",
    "DataLeakageError",
]
