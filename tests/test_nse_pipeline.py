"""TradeMind NSE Market Data Configuration, Downloader, Provenance, and Splitting Tests.

Deterministic offline test suite with mocked yfinance responses.
Tests configuration validation, MultiIndex columns, safe refresh, duplicate protection,
provenance metadata recording, and chronological train/val/test splitting.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from backend.app.data.exceptions import (
    DataValidationError,
    EmptyDataError,
    InvalidSymbolError,
    ProviderUnavailableError,
)
from backend.app.data.market_config import ConfigValidationError, MarketDataConfig
from backend.app.data.pipeline import MarketDataPipeline
from backend.app.data.provenance import ProvenanceMetadata
from backend.app.data.provider_yfinance import YFinanceProvider, flatten_yfinance_columns
from backend.app.data.schema import COL_CLOSE, COL_OPEN, COL_TIMESTAMP, REQUIRED_COLUMNS
from backend.app.data.split import chronological_split


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_nse_raw_df():
    """Deterministic 5-bar daily OHLCV dataset representing NSE equity trading."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({
        "Open": [2500.0, 2520.0, 2510.0, 2530.0, 2550.0],
        "High": [2530.0, 2540.0, 2535.0, 2560.0, 2570.0],
        "Low": [2490.0, 2505.0, 2500.0, 2520.0, 2540.0],
        "Close": [2515.0, 2510.0, 2525.0, 2555.0, 2565.0],
        "Volume": [1500000, 1800000, 1200000, 2100000, 1900000],
    }, index=dates)


@pytest.fixture
def multiindex_nse_df(sample_nse_raw_df):
    """Simulates yfinance MultiIndex column structure: ('Close', 'RELIANCE.NS')."""
    df = sample_nse_raw_df.copy()
    multi_cols = pd.MultiIndex.from_tuples([
        ("Open", "RELIANCE.NS"),
        ("High", "RELIANCE.NS"),
        ("Low", "RELIANCE.NS"),
        ("Close", "RELIANCE.NS"),
        ("Volume", "RELIANCE.NS"),
    ])
    df.columns = multi_cols
    return df


@pytest.fixture
def temp_env():
    """Isolated temporary directory tree for raw, processed, metadata, and config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        raw_dir = base / "raw"
        proc_dir = base / "processed"
        meta_dir = base / "metadata"
        conf_dir = base / "config"
        for d in [raw_dir, proc_dir, meta_dir, conf_dir]:
            d.mkdir(parents=True, exist_ok=True)
        yield {
            "base": base,
            "raw": raw_dir,
            "processed": proc_dir,
            "metadata": meta_dir,
            "config": conf_dir,
        }


# ==============================================================================
# Tests: Configuration Loading & Validation
# ==============================================================================

def test_load_valid_market_data_config(temp_env):
    """Test loading and validating a standard NSE configuration file."""
    config_file = Path("config/market_data_nse.json")
    assert config_file.exists(), "Central config file config/market_data_nse.json should exist."

    cfg = MarketDataConfig.load_from_file(config_file)
    assert cfg.market == "NSE"
    assert cfg.frequency == "1d"
    assert "RELIANCE.NS" in cfg.symbols
    assert "TCS.NS" in cfg.symbols
    assert cfg.date_range.start_date == "2016-01-01"
    assert cfg.date_range.end_date == "2025-12-31"
    assert cfg.split_proportions.train == 0.70
    assert cfg.split_proportions.validation == 0.20
    assert cfg.split_proportions.test == 0.10


def test_reject_invalid_date_range():
    """Test that end_date <= start_date or malformed dates are rejected."""
    with pytest.raises(ConfigValidationError) as exc:
        MarketDataConfig.from_dict({
            "date_range": {"start_date": "2025-01-01", "end_date": "2020-01-01"}
        })
    assert "earlier than end_date" in str(exc.value)

    with pytest.raises(ConfigValidationError):
        MarketDataConfig.from_dict({
            "date_range": {"start_date": "not-a-date", "end_date": "2025-01-01"}
        })


def test_reject_invalid_split_proportions():
    """Test split proportions not summing to 1.0 or out of bounds are rejected."""
    with pytest.raises(ConfigValidationError) as exc:
        MarketDataConfig.from_dict({
            "split_proportions": {"train": 0.50, "validation": 0.20, "test": 0.10}  # Sums to 0.80
        })
    assert "must sum to 1.0" in str(exc.value)

    with pytest.raises(ConfigValidationError):
        MarketDataConfig.from_dict({
            "split_proportions": {"train": -0.1, "validation": 0.6, "test": 0.5}
        })


def test_reject_invalid_symbols_in_config():
    """Test configuration rejects invalid ticker characters."""
    with pytest.raises(ConfigValidationError) as exc:
        MarketDataConfig.from_dict({"symbols": ["RELIANCE.NS", "INVALID$$$SYMBOL"]})
    assert "Invalid symbol" in str(exc.value)


# ==============================================================================
# Tests: Downloader & MultiIndex Column Handling
# ==============================================================================

def test_flatten_yfinance_multiindex_columns(multiindex_nse_df):
    """Test that MultiIndex columns from yfinance are correctly flattened to canonical names."""
    assert isinstance(multiindex_nse_df.columns, pd.MultiIndex)
    flat_df = flatten_yfinance_columns(multiindex_nse_df)
    assert not isinstance(flat_df.columns, pd.MultiIndex)
    assert list(flat_df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_mocked_yfinance_download_success(sample_nse_raw_df, temp_env):
    """Test downloading a symbol with mocked yfinance responses writes raw, processed, and metadata."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = sample_nse_raw_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest_path, report, prov = pipeline.download_and_process_symbol(
            symbol="RELIANCE.NS",
            start="2024-01-01",
            end="2024-01-06",
            interval="1d",
            auto_adjust=True,
            refresh=True,
        )

    # 1. Processed file checks
    assert dest_path.exists()
    assert "reliance_ns_1d_processed.csv" in dest_path.name
    assert report.valid_row_count == 5
    assert prov.download_status == "REFRESHED"
    assert prov.validation_status == "PASSED"

    # 2. Raw file checks (raw file created and untouched)
    raw_files = list(temp_env["raw"].glob("reliance_ns_1d_raw_*.csv"))
    assert len(raw_files) == 1
    assert raw_files[0].stat().st_size > 0

    # 3. Provenance metadata check
    meta_files = list(temp_env["metadata"].glob("reliance_ns_1d_provenance.json"))
    assert len(meta_files) == 1
    meta_json = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta_json["symbol"] == "RELIANCE.NS"
    assert meta_json["auto_adjust"] is True
    assert meta_json["row_count"] == 5


def test_empty_download_raises_empty_data_error(temp_env):
    """Test that empty yfinance responses are rejected and not treated as successful empty data."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()  # Empty dataframe

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(EmptyDataError) as exc:
            pipeline.download_and_process_symbol("TCS.NS", refresh=True)
        assert "0 records" in str(exc.value)


def test_network_failure_and_retry(temp_env):
    """Test that network errors trigger retries and ultimately raise ProviderUnavailableError."""
    provider = YFinanceProvider(max_retries=2, retry_backoff_factor=1.0)

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = ConnectionError("Simulated socket timeout")

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderUnavailableError) as exc:
            provider.fetch("INFY.NS")
        assert "after 2 attempts" in str(exc.value)


# ==============================================================================
# Tests: Safe Refresh & Duplicate Protection
# ==============================================================================

def test_safe_refresh_merges_without_duplication(temp_env):
    """Test refreshing dataset merges overlapping dates deterministically without row inflation."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )

    # Initial batch: 3 days (Jan 1 to Jan 3)
    dates_batch1 = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    batch1_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1000, 1000],
    }, index=dates_batch1)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = batch1_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest1, report1, _ = pipeline.download_and_process_symbol("SBIN.NS", refresh=True)
    assert report1.valid_row_count == 3

    # Refresh batch: overlaps Jan 3, adds Jan 4 and Jan 5 (with updated Close on Jan 3 = 108.0)
    dates_batch2 = pd.date_range("2024-01-03", periods=3, freq="D", tz="UTC")
    batch2_df = pd.DataFrame({
        "Open": [102.0, 103.0, 104.0],
        "High": [109.0, 108.0, 109.0],
        "Low": [101.0, 102.0, 103.0],
        "Close": [108.0, 107.0, 108.0],
        "Volume": [1500, 1000, 1000],
    }, index=dates_batch2)

    mock_ticker.history.return_value = batch2_df
    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest2, report2, prov2 = pipeline.download_and_process_symbol("SBIN.NS", refresh=True)

    # Overlap resolution: 3 days + 3 days with 1 overlap = exactly 5 unique chronological days
    assert report2.valid_row_count == 5
    merged_csv = pd.read_csv(dest2)
    assert len(merged_csv) == 5
    assert merged_csv["timestamp"].is_monotonic_increasing

    # Overwrite check on duplicate date (Jan 3) -> incoming close 108.0 retained
    jan3_row = merged_csv[merged_csv["timestamp"].str.startswith("2024-01-03")]
    assert len(jan3_row) == 1
    assert jan3_row.iloc[0]["close"] == 108.0


def test_failed_refresh_preserves_existing_data(temp_env):
    """Test that if a refresh download fails, previously validated data is left untouched."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )

    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    valid_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1000, 1000],
    }, index=dates)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = valid_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest_path, _, _ = pipeline.download_and_process_symbol("ICICIBANK.NS", refresh=True)

    original_bytes = dest_path.read_bytes()

    # Now simulate a subsequent failed refresh
    mock_ticker.history.side_effect = ConnectionError("Network dropped")
    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderUnavailableError):
            pipeline.download_and_process_symbol("ICICIBANK.NS", refresh=True)

    # Verify original processed data was NOT deleted or corrupted
    assert dest_path.exists()
    assert dest_path.read_bytes() == original_bytes


def test_provider_partial_download_insufficient_records(temp_env):
    """Test that partial downloads with insufficient bars are rejected and existing data is preserved."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )
    mock_ticker = MagicMock()
    # Mocking validation error when download has insufficient or empty records
    with patch("yfinance.Ticker", return_value=mock_ticker):
        with patch("backend.app.data.pipeline.validate_ohlcv_dataframe", side_effect=DataValidationError("Insufficient records in partition")):
            mock_ticker.history.return_value = pd.DataFrame({"Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [104.0], "Volume": [1000]}, index=pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"))
            with pytest.raises(DataValidationError) as exc:
                pipeline.download_and_process_symbol("TCS.NS", refresh=True)
            assert "Insufficient records" in str(exc.value)


def test_provider_invalid_ohlcv_values_rejected(temp_env):
    """Test that invalid OHLCV bars (e.g. high < low) in provider response are rejected while valid rows pass."""
    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    # Row 2 has high < low (90 < 105)
    bad_df = pd.DataFrame({
        "Open": [100.0, 100.0, 102.0],
        "High": [105.0, 90.0, 107.0],
        "Low": [99.0, 105.0, 101.0],
        "Close": [104.0, 95.0, 106.0],
        "Volume": [1000, 1000, 1000],
    }, index=dates)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = bad_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest_path, report, prov = pipeline.download_and_process_symbol("INFY.NS", refresh=True)

    assert report.rejected_row_count == 1
    assert report.valid_row_count == 2
    assert prov.validation_status == "WARNINGS"


def test_failed_write_permission_error_preserves_original_data(temp_env):
    """Test that if saving to processed directory encounters a permission error, DataIngestionError is raised and original data is preserved."""
    from backend.app.data.exceptions import DataIngestionError

    pipeline = MarketDataPipeline(
        raw_dir=temp_env["raw"],
        processed_dir=temp_env["processed"],
        metadata_dir=temp_env["metadata"],
    )
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    valid_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1000, 1000],
    }, index=dates)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = valid_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        dest_path, _, _ = pipeline.download_and_process_symbol("SBIN.NS", refresh=True)

    original_content = dest_path.read_text(encoding="utf-8")

    # Simulate locked file raising PermissionError on pandas to_csv when writing processed dataset
    orig_to_csv = pd.DataFrame.to_csv
    def mock_to_csv(self, path, *args, **kwargs):
        if "processed" in str(path):
            raise PermissionError("File locked by Excel")
        return orig_to_csv(self, path, *args, **kwargs)

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
            with pytest.raises(DataIngestionError) as exc:
                pipeline.download_and_process_symbol("SBIN.NS", refresh=True)
            assert "locked by another process" in str(exc.value)

    # Verify original file content was preserved
    assert dest_path.read_text(encoding="utf-8") == original_content


# ==============================================================================
# Tests: Chronological Dataset Splitting
# ==============================================================================

def test_chronological_split_strict_ordering():
    """Test 70/20/10 chronological split strictly preserves time order and isolation."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(111, 120, 100),
        "low": np.random.uniform(90, 99, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 5000, 100),
    })

    splits = chronological_split(df, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)

    # 1. Count checks (approx 70, 20, 10)
    assert splits.metadata.train_count == 70
    assert splits.metadata.val_count == 20
    assert splits.metadata.test_count == 10

    # 2. Strict temporal inequality: max(train) < min(val) < min(test)
    max_train_ts = splits.train["timestamp"].max()
    min_val_ts = splits.validation["timestamp"].min()
    max_val_ts = splits.validation["timestamp"].max()
    min_test_ts = splits.test["timestamp"].min()

    assert max_train_ts < min_val_ts
    assert max_val_ts < min_test_ts


def test_chronological_split_with_embargo():
    """Test embargo drops buffer bars to prevent rolling indicator lookahead across boundaries."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000
    })

    embargo = 5
    splits = chronological_split(df, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, embargo_bars=embargo)

    # Train had 70, drops 5 buffer bars -> 65
    assert splits.metadata.train_count == 65
    # Val had 20, drops 5 buffer bars -> 15
    assert splits.metadata.val_count == 15
    # Test retains full 10
    assert splits.metadata.test_count == 10
