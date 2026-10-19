"""TradeMind Phase 8 Market-Regime Detection Configuration.

Defines hyperparameters, feature selections, clustering parameters, and leakage
prevention bounds for the K-Means market-regime detection module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RegimeConfig:
    """Configuration hyperparameters for Market-Regime Detection.
    
    Attributes:
        n_clusters: Number of unsupervised regime clusters (default 3).
        feature_names: List of normalized feature names used for clustering.
                       Defaults to return, rolling volatility, and momentum.
        volatility_window: Rolling window length for return volatility (default 20 bars).
        momentum_period: Period for momentum oscillator (RSI / returns, default 14 bars).
        random_state: Deterministic random seed for reproducible K-Means initialization.
        n_init: Number of random initialization runs for K-Means (default 10).
        max_iter: Maximum number of iterations for K-Means convergence (default 300).
        scale_features: Whether to apply StandardScaler fitted strictly on training data (default True).
        impute_missing: Whether to apply SimpleImputer fitted strictly on training data (default True).
        train_ratio: Proportion allocated to chronological training split (default 0.70).
        val_ratio: Proportion allocated to chronological validation split (default 0.20).
        test_ratio: Proportion allocated to chronological test split (default 0.10).
        embargo_bars: Buffer bars dropped between splits to prevent indicator leakage (default 0).
        min_train_samples: Minimum valid training observations required to fit the model (default 30).
        model_version: Semantic version identifier for the regime model artifact.
    """
    n_clusters: int = 3
    feature_names: List[str] = field(
        default_factory=lambda: ["simple_return", "volatility_20", "momentum_rsi"]
    )
    volatility_window: int = 20
    momentum_period: int = 14
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300
    scale_features: bool = True
    impute_missing: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    embargo_bars: int = 0
    min_train_samples: int = 30
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be at least 2, got {self.n_clusters}")
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        if self.volatility_window < 2:
            raise ValueError(f"volatility_window must be at least 2, got {self.volatility_window}")
        if self.momentum_period < 2:
            raise ValueError(f"momentum_period must be at least 2, got {self.momentum_period}")
        if self.n_init < 1:
            raise ValueError(f"n_init must be at least 1, got {self.n_init}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars cannot be negative, got {self.embargo_bars}")
        if self.min_train_samples < self.n_clusters:
            raise ValueError(
                f"min_train_samples ({self.min_train_samples}) cannot be less than n_clusters ({self.n_clusters})"
            )

        total_ratio = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if abs(total_ratio - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
