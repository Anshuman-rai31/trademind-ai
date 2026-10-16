"""TradeMind Market Data Validator.

Implements rigorous data integrity verification:
1. Timestamp parsing, UTC normalization, and chronological sorting.
2. Duplicate timestamp detection and deterministic resolution.
3. Numeric type conversion and missing-value logging.
4. Positive price verification (open > 0, high > 0, low > 0, close > 0).
5. Non-negative volume verification (volume >= 0).
6. OHLC relationship checks (low <= open <= high, low <= close <= high, high >= low).
7. Time-series gap detection without synthetic filling.
8. Rejected row accounting and detailed diagnostic reporting.
"""
import re
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .exceptions import (
    DataValidationError,
    EmptyDataError,
    InsufficientDataError,
    InvalidSymbolError,
    SchemaValidationError,
)
from .models import RejectedRow, TimeSeriesGap, ValidationReport
from .schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_SOURCE,
    COL_SYMBOL,
    COL_TIMESTAMP,
    COL_VOLUME,
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
)

# Common interval mappings to seconds
INTERVAL_TO_SECONDS: Dict[str, float] = {
    "1m": 60.0,
    "2m": 120.0,
    "5m": 300.0,
    "15m": 900.0,
    "30m": 1800.0,
    "1h": 3600.0,
    "2h": 7200.0,
    "4h": 14400.0,
    "1d": 86400.0,
    "1D": 86400.0,
    "1w": 604800.0,
    "1W": 604800.0,
}


def parse_interval_to_seconds(interval: str) -> Optional[float]:
    """Parse an interval string (e.g. '1m', '15m', '1h', '1d') into seconds."""
    if not interval:
        return None
    cleaned = interval.strip()
    if cleaned in INTERVAL_TO_SECONDS:
        return INTERVAL_TO_SECONDS[cleaned]

    # Regex fallback for custom intervals like '10m', '3h', '2d'
    match = re.match(r"^(\d+)\s*([mshdwMSHDW])$", cleaned)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower()
        multipliers = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
            "w": 604800.0,
        }
        return val * multipliers.get(unit, 60.0)
    return None


def validate_symbol(symbol: str) -> str:
    """Validate that an asset symbol conforms to safe alphanumeric patterns.
    
    Args:
        symbol: Asset ticker symbol.
        
    Returns:
        Cleaned uppercase symbol.
        
    Raises:
        InvalidSymbolError: If symbol is empty or contains unsafe characters.
    """
    if not symbol or not isinstance(symbol, str):
        raise InvalidSymbolError("Symbol must be a non-empty string.")
    cleaned = symbol.strip().upper()
    # Allow alphanumeric characters, dot, dash, underscore (e.g., BRK.B, BTC-USD)
    if not re.match(r"^[A-Z0-9.\-_]{1,30}$", cleaned):
        raise InvalidSymbolError(
            f"Invalid asset symbol '{symbol}'. Allowed characters: A-Z, 0-9, '.', '-', '_'"
        )
    return cleaned


def parse_and_normalize_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps and normalize strictly to UTC ISO 8601 strings.
    
    Timezone Convention:
        - Naive timestamps are localized to UTC.
        - Timezone-aware timestamps are converted to UTC.
        - Unparseable timestamps are coerced to NaT for rejection accounting.
        
    Args:
        series: Raw timestamp values from dataframe.
        
    Returns:
        pd.Series of pd.Timestamp objects in UTC.
    """
    # Coerce errors to NaT
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed


def detect_series_gaps(
    timestamps: pd.Series,
    interval_str: Optional[str] = None
) -> List[TimeSeriesGap]:
    """Identify time series gaps between consecutive timestamps without filling.
    
    Args:
        timestamps: Chronologically sorted pandas Series of UTC timestamps.
        interval_str: Expected interval (e.g., '1d', '1h', '1m').
        
    Returns:
        List of TimeSeriesGap diagnostic objects.
    """
    if len(timestamps) < 2:
        return []

    expected_seconds = parse_interval_to_seconds(interval_str) if interval_str else None

    # If no interval is provided, infer from median delta if sufficient records
    if expected_seconds is None and len(timestamps) >= 3:
        deltas = timestamps.diff().dropna().dt.total_seconds()
        median_delta = deltas.median()
        if median_delta > 0:
            expected_seconds = median_delta

    if expected_seconds is None or expected_seconds <= 0:
        return []

    # Threshold: an interval larger than 1.5x expected step is a gap
    gap_threshold = expected_seconds * 1.5
    gaps: List[TimeSeriesGap] = []

    for i in range(1, len(timestamps)):
        prev_t = timestamps.iloc[i - 1]
        curr_t = timestamps.iloc[i]
        delta_sec = (curr_t - prev_t).total_seconds()

        if delta_sec > gap_threshold:
            # Estimate missing bars
            missing_bars = max(1, int(round(delta_sec / expected_seconds)) - 1)
            gaps.append(TimeSeriesGap(
                start_timestamp=prev_t.isoformat(),
                end_timestamp=curr_t.isoformat(),
                duration_seconds=delta_sec,
                expected_interval=interval_str or f"{expected_seconds}s",
                estimated_missing_bars=missing_bars,
            ))

    return gaps


def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    source: str = "csv",
    interval: Optional[str] = "1d",
    min_records: int = 1,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Perform rigorous validation and normalization on an OHLCV dataframe.
    
    Data Integrity Rules Enforced:
        1. All required OHLCV columns must be present.
        2. Timestamps normalized to UTC and sorted chronologically.
        3. Duplicates detected; last occurrence retained; duplicates reported.
        4. Prices must be strictly positive (open > 0, high > 0, low > 0, close > 0).
        5. Volume must be non-negative (volume >= 0).
        6. Sensible OHLC relationships:
           - low <= open <= high
           - low <= close <= high
           - high >= low
        7. Gaps detected and reported, but NEVER silently filled.
        8. Rejected rows documented with exact reasons and indices.
        
    Args:
        df: Input DataFrame with raw data.
        symbol: Asset symbol (optional, will be validated if provided).
        source: Source description (e.g. 'csv', 'yfinance').
        interval: Expected bar frequency (e.g. '1d', '1h', '1m').
        min_records: Minimum acceptable valid rows (defaults to 1).
        
    Returns:
        Tuple of (clean_validated_dataframe, ValidationReport).
        
    Raises:
        EmptyDataError: If input dataframe has 0 rows.
        SchemaValidationError: If required columns are missing.
        InsufficientDataError: If remaining valid rows < min_records.
        DataValidationError: If all rows are invalid or rejected.
    """
    if df is None or df.empty:
        raise EmptyDataError("Input dataset is empty or contains no records.")

    raw_count = len(df)
    clean_symbol = validate_symbol(symbol) if symbol else "UNKNOWN"

    report = ValidationReport(
        symbol=clean_symbol,
        source=source,
        interval=interval,
        raw_row_count=raw_count,
    )

    # 1. Verify required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Missing required OHLCV columns: {missing_cols}. "
            f"Required schema: {REQUIRED_COLUMNS}"
        )

    # Work on a copy to guarantee no mutation of raw source data
    work_df = df.copy()

    # 2. Parse & normalize timestamps to UTC
    parsed_timestamps = parse_and_normalize_timestamps(work_df[COL_TIMESTAMP])
    work_df[COL_TIMESTAMP] = parsed_timestamps

    # 3. Numeric conversion of price and volume columns
    for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    # 4. Row-level validity evaluation
    rejected_rows: List[RejectedRow] = []
    valid_indices: List[int] = []

    for idx, row in work_df.iterrows():
        reasons: List[str] = []
        raw_row_dict = {k: row[k] for k in REQUIRED_COLUMNS if k in row}

        # Check timestamp
        ts = row[COL_TIMESTAMP]
        if pd.isna(ts):
            reasons.append("Invalid or unparseable timestamp")
            ts_str = None
        else:
            ts_str = ts.isoformat()

        # Check numeric nulls
        null_cols = [c for c in REQUIRED_COLUMNS if pd.isna(row[c])]
        if null_cols:
            reasons.append(f"Missing/NaN values in: {', '.join(null_cols)}")
            report.missing_value_count += len(null_cols)

        # If not null, check price positivity
        if not any(pd.isna(row[c]) for c in PRICE_COLUMNS):
            o, h, l, c = row[COL_OPEN], row[COL_HIGH], row[COL_LOW], row[COL_CLOSE]
            
            if o <= 0:
                reasons.append(f"Non-positive open price: {o}")
            if h <= 0:
                reasons.append(f"Non-positive high price: {h}")
            if l <= 0:
                reasons.append(f"Non-positive low price: {l}")
            if c <= 0:
                reasons.append(f"Non-positive close price: {c}")

            # Relational checks
            if h < l:
                reasons.append(f"OHLC violation: high ({h}) < low ({l})")
            if o < l or o > h:
                reasons.append(f"OHLC violation: open ({o}) outside low-high range [{l}, {h}]")
            if c < l or c > h:
                reasons.append(f"OHLC violation: close ({c}) outside low-high range [{l}, {h}]")

        # Check volume non-negativity
        if not pd.isna(row[COL_VOLUME]):
            v = row[COL_VOLUME]
            if v < 0:
                reasons.append(f"Negative volume: {v}")

        if reasons:
            rejected_rows.append(RejectedRow(
                index=int(idx),
                timestamp=ts_str,
                reasons=reasons,
                raw_values=raw_row_dict,
            ))
        else:
            valid_indices.append(idx)

    report.rejected_rows = rejected_rows
    report.rejected_row_count = len(rejected_rows)

    if not valid_indices:
        raise DataValidationError(
            f"All {raw_count} rows in dataset failed validation checks. "
            f"See sample rejection reasons: {[r.reasons for r in rejected_rows[:3]]}"
        )

    # Filter to valid records
    valid_df = work_df.loc[valid_indices].copy()

    # 5. Sort chronologically
    valid_df.sort_values(by=COL_TIMESTAMP, ascending=True, inplace=True)

    # 6. Detect duplicate timestamps
    dup_mask = valid_df.duplicated(subset=[COL_TIMESTAMP], keep=False)
    if dup_mask.any():
        dup_series = valid_df.loc[dup_mask, COL_TIMESTAMP]
        dup_list = [t.isoformat() for t in dup_series.unique()]
        report.duplicate_timestamp_count = len(valid_df[dup_mask]) - len(dup_list)
        report.duplicate_timestamps = dup_list
        # Deduplicate deterministically: keep last occurrence
        valid_df.drop_duplicates(subset=[COL_TIMESTAMP], keep="last", inplace=True)

    # 7. Check remaining record count against minimum threshold
    final_count = len(valid_df)
    report.valid_row_count = final_count

    if final_count < min_records:
        raise InsufficientDataError(
            f"Dataset contains {final_count} valid rows, which is below the minimum "
            f"required threshold of {min_records} records."
        )

    # 8. Time range metadata
    start_ts = valid_df[COL_TIMESTAMP].iloc[0]
    end_ts = valid_df[COL_TIMESTAMP].iloc[-1]
    report.start_timestamp = start_ts.isoformat()
    report.end_timestamp = end_ts.isoformat()

    # 9. Time series gap detection
    report.detected_gaps = detect_series_gaps(valid_df[COL_TIMESTAMP], interval)

    # 10. Attach metadata columns
    valid_df[COL_SYMBOL] = clean_symbol
    valid_df[COL_SOURCE] = source

    # Standardize column order
    final_cols = REQUIRED_COLUMNS + [COL_SYMBOL, COL_SOURCE]
    clean_output_df = valid_df[final_cols].reset_index(drop=True)

    return clean_output_df, report
