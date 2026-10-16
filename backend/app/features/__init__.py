"""TradeMind Feature Engineering Package (Phase 3).

Provides reusable, stateless feature engineering functions for financial
time-series data. All functions operate on validated OHLCV DataFrames and
return new DataFrames with additional feature columns appended.

Design Principles:
    - Stateless: Every function is a pure transformation; no in-place mutation.
    - Composable: apply_all_features() chains individual functions via pipeline.
    - NaN-safe: Leading NaN rows arising from warm-up windows are preserved
      and documented so consumers (e.g. RL environments) can drop or mask them.
    - No lookahead: All calculations use only past or current bar data.
    - Tested: Every function covered by deterministic unit tests.
"""

from .engineer import (
    # Individual feature groups
    compute_returns,
    compute_moving_averages,
    compute_rsi,
    compute_bollinger_bands,
    compute_macd,
    compute_atr,
    compute_obv,
    compute_normalization,
    # High-level pipeline
    apply_all_features,
    # Feature config
    FeatureConfig,
)
from .report import FeatureReport, generate_feature_report

__all__ = [
    # Individual Feature Computations
    "compute_returns",
    "compute_moving_averages",
    "compute_rsi",
    "compute_bollinger_bands",
    "compute_macd",
    "compute_atr",
    "compute_obv",
    "compute_normalization",
    # High-level Pipeline
    "apply_all_features",
    # Configuration
    "FeatureConfig",
    # Reporting
    "FeatureReport",
    "generate_feature_report",
]
