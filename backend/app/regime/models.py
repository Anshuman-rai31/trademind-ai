"""TradeMind Phase 8 Market-Regime Output & Summary Data Structures.

Canonical dataclasses holding cluster profiles, descriptive historical statistics,
regime audit metadata, and output datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ClusterSummary:
    """Historical descriptive profile for an individual unsupervised regime cluster.
    
    CRITICAL DISCLAIMER:
        These statistics are historical training-period descriptive characterizations,
        NOT guarantees or directional predictions of future returns.
    
    Attributes:
        cluster_id: Arbitrary integer identifier assigned by K-Means.
        count: Number of training observations belonging to this cluster.
        pct_observations: Fraction of total training dataset in this cluster.
        mean_return: Arithmetic mean daily return of the cluster.
        median_return: Median daily return of the cluster.
        mean_volatility: Average rolling volatility level of the cluster.
        mean_momentum: Average momentum indicator (e.g. RSI) of the cluster.
        interpreted_label: Human-readable descriptive tag based on empirical characteristics.
        symbol_distribution: Count of observations per symbol assigned to this cluster.
    """
    cluster_id: int
    count: int
    pct_observations: float
    mean_return: float
    median_return: float
    mean_volatility: float
    mean_momentum: float
    interpreted_label: str
    symbol_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary with rounded metrics."""
        return {
            "cluster_id": self.cluster_id,
            "count": self.count,
            "pct_observations": round(self.pct_observations, 4),
            "mean_return": round(self.mean_return, 6),
            "median_return": round(self.median_return, 6),
            "mean_volatility": round(self.mean_volatility, 6),
            "mean_momentum": round(self.mean_momentum, 4),
            "interpreted_label": self.interpreted_label,
            "symbol_distribution": self.symbol_distribution,
            "disclaimer": "Historical descriptive statistics only; not a future prediction.",
        }


@dataclass
class RegimeModelMetadata:
    """Audit provenance documenting model training configuration and temporal boundaries."""
    model_version: str
    n_clusters: int
    feature_names: List[str]
    random_state: int
    training_sample_count: int
    train_start: Optional[str]
    train_end: Optional[str]
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    cluster_summaries: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


@dataclass
class RegimeOutput:
    """Comprehensive container for regime detection execution results.
    
    Attributes:
        labeled_df: Independent output DataFrame with assigned regime labels,
                    preserving all original source columns intact.
        train_df: Chronological training partition with regime labels.
        val_df: Chronological validation partition with regime labels.
        test_df: Chronological final test partition with regime labels.
        metadata: Auditable provenance metadata.
        cluster_summaries: List of ClusterSummary instances based on training data.
    """
    labeled_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: RegimeModelMetadata
    cluster_summaries: List[ClusterSummary]
