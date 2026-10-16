"""TradeMind Feature Engineering Report (Phase 3).

Provides FeatureReport — a structured summary of feature engineering output,
including NaN counts per column, warm-up analysis, and usable row count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FeatureReport:
    """Structured summary of an apply_all_features() call.

    Attributes:
        symbol: Ticker symbol the features were computed for.
        total_rows: Total number of rows in the output DataFrame.
        feature_columns: List of new feature column names added.
        base_columns: List of original OHLCV columns.
        nan_counts: Mapping of column -> count of NaN values.
        warmup_bars: Number of leading rows expected to have NaN features.
        usable_rows: Rows with a complete feature set (total_rows - warmup_bars).
        has_warnings: True if any feature column has unexpected NaN beyond warmup.
        warnings: List of diagnostic warning strings.
    """
    symbol: str
    total_rows: int
    feature_columns: List[str]
    base_columns: List[str]
    nan_counts: Dict[str, int]
    warmup_bars: int
    usable_rows: int
    has_warnings: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "total_rows": self.total_rows,
            "usable_rows": self.usable_rows,
            "warmup_bars": self.warmup_bars,
            "feature_columns": self.feature_columns,
            "base_columns": self.base_columns,
            "nan_counts": self.nan_counts,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        lines = [
            f"FeatureReport [{self.symbol}]",
            f"  Total rows    : {self.total_rows}",
            f"  Usable rows   : {self.usable_rows}",
            f"  Warmup bars   : {self.warmup_bars}",
            f"  Features added: {len(self.feature_columns)}",
        ]
        if self.has_warnings:
            lines.append(f"  ⚠ Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def generate_feature_report(
    original_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    symbol: str,
    warmup_bars: int,
) -> FeatureReport:
    """Generate a diagnostic FeatureReport comparing pre- and post-feature DataFrames.

    Args:
        original_df: The original OHLCV DataFrame before feature engineering.
        featured_df: The augmented DataFrame returned by apply_all_features().
        symbol: Ticker symbol identifier for the report.
        warmup_bars: Expected number of leading rows with NaN features.

    Returns:
        FeatureReport with NaN analysis, usable row count, and warnings.
    """
    base_cols = list(original_df.columns)
    feature_cols = [c for c in featured_df.columns if c not in base_cols]

    nan_counts: Dict[str, int] = {}
    warnings: List[str] = []

    for col in featured_df.columns:
        n_nan = int(featured_df[col].isna().sum())
        if n_nan > 0:
            nan_counts[col] = n_nan

    # Check for unexpected NaN in feature columns beyond warmup window
    for col in feature_cols:
        n_nan = nan_counts.get(col, 0)
        if n_nan > warmup_bars + 1:  # +1 tolerance for boundary edge cases
            warnings.append(
                f"Column '{col}' has {n_nan} NaN values, "
                f"exceeding expected warmup ({warmup_bars})."
            )

    usable = max(0, len(featured_df) - warmup_bars)

    return FeatureReport(
        symbol=symbol,
        total_rows=len(featured_df),
        feature_columns=feature_cols,
        base_columns=base_cols,
        nan_counts=nan_counts,
        warmup_bars=warmup_bars,
        usable_rows=usable,
        has_warnings=bool(warnings),
        warnings=warnings,
    )
