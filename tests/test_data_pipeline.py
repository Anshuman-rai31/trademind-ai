"""TradeMind Phase 2 Unit Tests: Historical Market Data Pipeline.

Comprehensive, deterministic test suite operating completely offline without
any external internet or live network dependencies.
"""
import io
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.app.data.exceptions import (
    DataFileNotFoundError,
    DataIngestionError,
    DataValidationError,
    EmptyDataError,
    InsufficientDataError,
    InvalidSymbolError,
    SchemaValidationError,
)
from backend.app.data.loader_csv import CSVDataLoader, load_ohlcv_csv
from backend.app.data.pipeline import MarketDataPipeline
from backend.app.data.provider_yfinance import YFinanceProvider
from backend.app.data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_SOURCE,
    COL_SYMBOL,
    COL_TIMESTAMP,
    COL_VOLUME,
    REQUIRED_COLUMNS,
)
from backend.app.data.validator import (
    detect_series_gaps,
    parse_and_normalize_timestamps,
    validate_ohlcv_dataframe,
    validate_symbol,
)


# ==============================================================================
# Deterministic Fixtures
# ==============================================================================

@pytest.fixture
def valid_csv_content():
    """A clean, 5-bar daily OHLCV dataset."""
    return (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000000\n"
        "2024-01-03,104.0,106.5,103.0,105.5,1200000\n"
        "2024-01-04,105.0,107.0,104.5,106.0,950000\n"
        "2024-01-05,106.0,108.0,105.0,107.5,1100000\n"
        "2024-01-08,108.0,110.0,107.0,109.0,1300000\n"
    )


@pytest.fixture
def temp_data_dirs():
    """Creates isolated temporary raw and processed directories for testing."""
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as proc_dir:
        yield Path(raw_dir), Path(proc_dir)


# ==============================================================================
# Tests: Loader & Schema Verification
# ==============================================================================

def test_load_valid_csv(valid_csv_content):
    """Test loading valid CSV content parses normalized lowercase column headers."""
    buffer = io.StringIO(valid_csv_content)
    df = CSVDataLoader.load(buffer, symbol="TEST")
    
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert len(df) == 5


def test_load_missing_file_raises_error():
    """Test attempting to load a non-existent file raises DataFileNotFoundError."""
    with pytest.raises(DataFileNotFoundError):
        CSVDataLoader.load("path/that/does/not/exist_12345.csv")


def test_load_empty_file_raises_error(temp_data_dirs):
    """Test loading a zero-byte file raises EmptyDataError."""
    raw_dir, _ = temp_data_dirs
    empty_file = raw_dir / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDataError):
        CSVDataLoader.load(empty_file)


def test_load_header_only_csv_raises_error():
    """Test loading a CSV with headers but no records raises EmptyDataError."""
    buffer = io.StringIO("Date,Open,High,Low,Close,Volume\n")
    with pytest.raises(EmptyDataError):
        CSVDataLoader.load(buffer)


def test_load_missing_required_column():
    """Test loading a CSV missing a mandatory column raises SchemaValidationError."""
    incomplete_csv = (
        "Date,Open,High,Low,Volume\n"  # Missing Close
        "2024-01-02,100.0,105.0,99.0,1000000\n"
    )
    buffer = io.StringIO(incomplete_csv)
    with pytest.raises(SchemaValidationError) as exc_info:
        CSVDataLoader.load(buffer)
    assert "close" in str(exc_info.value)


# ==============================================================================
# Tests: Timestamp Normalization, Sorting, & Duplicates
# ==============================================================================

def test_timestamp_normalization_and_sorting():
    """Test out-of-order timestamps with timezones are converted to UTC and sorted."""
    raw_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-03 14:30:00-05:00,102.0,104.0,101.0,103.0,1000\n"
        "2024-01-01 09:30:00-05:00,100.0,101.0,99.0,100.5,1000\n"
        "2024-01-02 09:30:00-05:00,101.0,103.0,100.0,102.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(raw_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST", interval="1d")

    # Verify chronological sorting
    ts_list = list(clean_df[COL_TIMESTAMP])
    assert ts_list[0] < ts_list[1] < ts_list[2]
    # Verify UTC normalization: 09:30:00-05:00 -> 14:30:00 UTC
    assert ts_list[0].hour == 14
    assert str(ts_list[0].tz) == "UTC"


def test_duplicate_timestamp_detection():
    """Test duplicate timestamps are detected, recorded in the report, and deduplicated."""
    dup_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
        "2024-01-02,100.0,106.0,99.0,105.0,2000\n"  # Duplicate date with updated close
        "2024-01-03,105.0,107.0,104.0,106.0,1500\n"
    )
    df = CSVDataLoader.load(io.StringIO(dup_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST")

    assert report.has_duplicates is True
    assert report.duplicate_timestamp_count == 1
    assert len(clean_df) == 2
    # Deterministic resolution: last occurrence kept (close=105.0)
    assert clean_df.iloc[0][COL_CLOSE] == 105.0


# ==============================================================================
# Tests: Missing Values, Invalid Prices, & Volume Checks
# ==============================================================================

def test_missing_values_rejection():
    """Test rows with missing/null values in numeric columns are rejected and reported."""
    dirty_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
        "2024-01-03,104.0,NaN,102.0,103.0,1000\n"     # Missing High
        "2024-01-04,103.0,105.0,101.0,104.0,NaN\n"    # Missing Volume
        "2024-01-05,104.0,106.0,103.0,105.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(dirty_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST")

    assert report.rejected_row_count == 2
    assert report.valid_row_count == 2
    assert len(clean_df) == 2
    assert report.missing_value_count == 2


def test_non_positive_prices_rejection():
    """Test negative or zero prices are rejected with descriptive reasons."""
    bad_price_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
        "2024-01-03,0.0,105.0,0.0,104.0,1000\n"       # Zero Open
        "2024-01-04,100.0,105.0,-10.0,104.0,1000\n"   # Negative Low
        "2024-01-05,100.0,105.0,99.0,104.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(bad_price_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST")

    assert report.rejected_row_count == 2
    assert report.valid_row_count == 2
    reasons = [r.reasons for r in report.rejected_rows]
    assert any("Non-positive open price: 0.0" in r for r in reasons)
    assert any("Non-positive low price: -10.0" in r for r in reasons)


def test_negative_volume_rejection():
    """Test negative volume is rejected with descriptive reason."""
    bad_volume_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
        "2024-01-03,104.0,106.0,103.0,105.0,-500\n"   # Negative volume
        "2024-01-04,105.0,107.0,104.0,106.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(bad_volume_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST")

    assert report.rejected_row_count == 1
    assert "Negative volume" in report.rejected_rows[0].reasons[0]
    assert "-500" in report.rejected_rows[0].reasons[0]


# ==============================================================================
# Tests: OHLC Relationship Consistency
# ==============================================================================

def test_ohlc_relational_violations():
    """Test relational checks: high < low, open outside [low, high], close outside [low, high]."""
    violations_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,95.0,105.0,102.0,1000\n"    # High < Low (95 < 105)
        "2024-01-03,110.0,105.0,99.0,102.0,1000\n"    # Open > High (110 > 105)
        "2024-01-04,100.0,105.0,99.0,95.0,1000\n"     # Close < Low (95 < 99)
        "2024-01-05,100.0,105.0,99.0,104.0,1000\n"    # Valid
    )
    df = CSVDataLoader.load(io.StringIO(violations_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST")

    assert report.rejected_row_count == 3
    assert report.valid_row_count == 1

    # Check reason messages
    all_reasons = " ".join([" ".join(r.reasons) for r in report.rejected_rows])
    assert "high (95.0) < low (105.0)" in all_reasons
    assert "open (110.0) outside low-high range" in all_reasons
    assert "close (95.0) outside low-high range" in all_reasons


# ==============================================================================
# Tests: Gap Detection Without Silent Filling
# ==============================================================================

def test_time_series_gap_detection_without_filling():
    """Test time series gaps are detected and reported, without synthetic rows created."""
    # Day 2, Day 3, then jumps to Day 10 (7 day gap)
    gap_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
        "2024-01-03,104.0,106.0,103.0,105.0,1000\n"
        "2024-01-10,105.0,107.0,104.0,106.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(gap_csv))
    clean_df, report = validate_ohlcv_dataframe(df, symbol="TEST", interval="1d")

    # Data integrity: length must remain exactly 3 (NO synthetic forward-filling)
    assert len(clean_df) == 3
    assert report.has_gaps is True
    assert len(report.detected_gaps) == 1

    gap = report.detected_gaps[0]
    assert "2024-01-03" in gap.start_timestamp
    assert "2024-01-10" in gap.end_timestamp
    assert gap.estimated_missing_bars >= 6


# ==============================================================================
# Tests: Edge Cases & Error Handling
# ==============================================================================

def test_all_rows_invalid_raises_data_validation_error():
    """Test when all rows fail validation, DataValidationError is raised."""
    all_bad_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,-10.0,105.0,99.0,104.0,1000\n"
        "2024-01-03,100.0,90.0,110.0,104.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(all_bad_csv))
    with pytest.raises(DataValidationError):
        validate_ohlcv_dataframe(df, symbol="TEST")


def test_insufficient_data_error():
    """Test when valid rows fall below min_records threshold."""
    one_row_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100.0,105.0,99.0,104.0,1000\n"
    )
    df = CSVDataLoader.load(io.StringIO(one_row_csv))
    with pytest.raises(InsufficientDataError):
        validate_ohlcv_dataframe(df, symbol="TEST", min_records=2)


def test_invalid_symbol_rejection():
    """Test unsafe symbols containing path traversal or invalid characters are rejected."""
    with pytest.raises(InvalidSymbolError):
        validate_symbol("../bad/symbol")
    with pytest.raises(InvalidSymbolError):
        validate_symbol("$$$INVALID")
    with pytest.raises(InvalidSymbolError):
        validate_symbol("")

    # Valid symbols must pass
    assert validate_symbol("aapl") == "AAPL"
    assert validate_symbol("BTC-USD") == "BTC-USD"
    assert validate_symbol("BRK.B") == "BRK.B"


# ==============================================================================
# Tests: End-to-End Pipeline & Raw File Immutability
# ==============================================================================

def test_raw_file_immutability(valid_csv_content, temp_data_dirs):
    """Test processing a raw CSV file never alters or touches the raw input file."""
    raw_dir, proc_dir = temp_data_dirs
    raw_file = raw_dir / "aapl_raw.csv"
    raw_file.write_text(valid_csv_content, encoding="utf-8")

    # Read original contents
    original_bytes = raw_file.read_bytes()

    pipeline = MarketDataPipeline(raw_dir=raw_dir, processed_dir=proc_dir)
    dest_path, report = pipeline.process_csv(raw_file, symbol="AAPL", interval="1d")

    # 1. Raw file must be 100% byte-for-byte identical
    assert raw_file.read_bytes() == original_bytes

    # 2. Processed file exists in processed directory
    assert dest_path.exists()
    assert dest_path.parent == proc_dir
    assert "aapl_1d_" in dest_path.name

    # 3. Companion report exists
    json_report = dest_path.with_suffix(".json")
    assert json_report.exists()
    report_dict = json.loads(json_report.read_text(encoding="utf-8"))
    assert report_dict["symbol"] == "AAPL"
    assert report_dict["valid_row_count"] == 5


def test_yfinance_provider_graceful_handling():
    """Test YFinanceProvider handles missing symbol / empty response gracefully."""
    provider = YFinanceProvider()
    # Provider availability flag should be a boolean
    assert isinstance(provider.is_available, bool)

    # When provider is given invalid symbol, it raises InvalidSymbolError
    with pytest.raises(InvalidSymbolError):
        provider.fetch("INVALID$$$NAME")
