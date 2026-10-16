"""TradeMind Chronological Dataset Splitting Utility.

Implements strict temporal train/validation/test partitioning for financial time-series.
Ensures zero future-data leakage into earlier datasets and supports embargo gaps
to prevent lookahead leakage across overlapping rolling indicator windows.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd

from .exceptions import DataValidationError
from .schema import COL_TIMESTAMP


@dataclass
class SplitMetadata:
    """Documents exact temporal intervals and sample sizes for chronological partitions."""
    train_count: int
    val_count: int
    test_count: int
    embargo_bars: int
    train_start: Optional[str]
    train_end: Optional[str]
    val_start: Optional[str]
    val_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]
    ratios: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "embargo_bars": self.embargo_bars,
            "train_interval": [self.train_start, self.train_end],
            "val_interval": [self.val_start, self.val_end],
            "test_interval": [self.test_start, self.test_end],
            "ratios": self.ratios,
        }


@dataclass
class ChronologicalSplits:
    """Container holding strictly partitioned DataFrames and provenance metadata."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: SplitMetadata


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    timestamp_col: str = COL_TIMESTAMP,
    embargo_bars: int = 0,
) -> ChronologicalSplits:
    """Split a market dataset chronologically into Train, Validation, and Test partitions.
    
    Data Integrity Guarantees:
        1. Shuffling is strictly prohibited.
        2. Strict temporal ordering: max(train_date) < min(val_date) < min(test_date).
        3. Optional embargo_bars drops k observations immediately preceding partition boundaries
           to guarantee overlapping rolling indicator windows (e.g. 20-day SMA, 14-day RSI)
           cannot leak information from validation into training or test into validation.
           
    Args:
        df: Input DataFrame with validated historical OHLCV data.
        train_ratio: Proportion allocated to training (default 0.70).
        val_ratio: Proportion allocated to validation (default 0.20).
        test_ratio: Proportion allocated to final test (default 0.10).
        timestamp_col: Name of the timestamp column (default 'timestamp').
        embargo_bars: Number of buffer bars to purge before partition transitions.
        
    Returns:
        ChronologicalSplits dataclass with train, validation, test DataFrames and metadata.
        
    Raises:
        DataValidationError: If dataset is too small or ratios are invalid.
    """
    if df is None or df.empty:
        raise DataValidationError("Cannot split empty DataFrame.")

    if timestamp_col not in df.columns:
        raise DataValidationError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    total_ratio = round(train_ratio + val_ratio + test_ratio, 6)
    if abs(total_ratio - 1.0) > 1e-5:
        raise DataValidationError(f"Split ratios must sum to 1.0 (got {total_ratio}).")

    # Sort strictly chronologically
    sorted_df = df.sort_values(by=timestamp_col, ascending=True).reset_index(drop=True)
    n = len(sorted_df)

    min_required = 10 + 2 * embargo_bars
    if n < min_required:
        raise DataValidationError(
            f"Dataset has {n} rows, which is insufficient for 3-way chronological split "
            f"with embargo={embargo_bars} (minimum required: {min_required})."
        )

    # Calculate index boundaries
    train_end_idx = int(round(n * train_ratio))
    val_end_idx = int(round(n * (train_ratio + val_ratio)))

    # Ensure at least 1 row in every set
    train_end_idx = max(1, min(train_end_idx, n - 2))
    val_end_idx = max(train_end_idx + 1, min(val_end_idx, n - 1))

    # Apply embargo buffers if requested
    # Drop embargo_bars immediately preceding the split transitions
    train_slice_end = max(1, train_end_idx - embargo_bars) if embargo_bars > 0 else train_end_idx
    val_slice_end = max(train_end_idx + 1, val_end_idx - embargo_bars) if embargo_bars > 0 else val_end_idx

    train_df = sorted_df.iloc[:train_slice_end].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[train_end_idx:val_slice_end].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[val_end_idx:].copy().reset_index(drop=True)

    def get_iso(series: pd.Series, idx: int) -> Optional[str]:
        if series.empty:
            return None
        val = series.iloc[idx]
        return val.isoformat() if hasattr(val, "isoformat") else str(val)

    metadata = SplitMetadata(
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        embargo_bars=embargo_bars,
        train_start=get_iso(train_df[timestamp_col], 0) if len(train_df) else None,
        train_end=get_iso(train_df[timestamp_col], -1) if len(train_df) else None,
        val_start=get_iso(val_df[timestamp_col], 0) if len(val_df) else None,
        val_end=get_iso(val_df[timestamp_col], -1) if len(val_df) else None,
        test_start=get_iso(test_df[timestamp_col], 0) if len(test_df) else None,
        test_end=get_iso(test_df[timestamp_col], -1) if len(test_df) else None,
        ratios={"train": train_ratio, "validation": val_ratio, "test": test_ratio},
    )

    return ChronologicalSplits(
        train=train_df,
        validation=val_df,
        test=test_df,
        metadata=metadata,
    )
