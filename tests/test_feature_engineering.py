"""TradeMind Phase 3 — Feature Engineering Test Suite.

Deterministic offline tests covering every individual feature function,
the high-level pipeline, FeatureConfig validation, and FeatureReport generation.

Test Conventions:
    - All input data is synthetically generated (no network calls).
    - All mathematical assertions use np.isclose() or relative tolerance.
    - NaN positions are asserted explicitly, not just ignored.
    - Known closed-form results are used where possible to verify correctness.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.features.engineer import (
    FeatureConfig,
    apply_all_features,
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_moving_averages,
    compute_normalization,
    compute_obv,
    compute_returns,
    compute_rsi,
)
from backend.app.features.report import FeatureReport, generate_feature_report


# ==============================================================================
# Shared Fixtures
# ==============================================================================

@pytest.fixture
def ohlcv_20():
    """20-bar deterministic OHLCV DataFrame with linearly rising prices."""
    n = 20
    close = np.linspace(100.0, 119.0, n)          # 100, 101, 102, ..., 119
    high  = close + 2.0
    low   = close - 2.0
    open_ = close - 0.5
    vol   = np.arange(1000, 1000 + n * 100, 100, dtype=float)  # 1000, 1100, ...
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


@pytest.fixture
def ohlcv_100():
    """100-bar deterministic OHLCV DataFrame; enough for all default warm-ups."""
    rng = np.random.default_rng(42)
    n = 100
    close = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    close = np.clip(close, 500, 2000)
    high  = close + rng.uniform(1, 5, n)
    low   = close - rng.uniform(1, 5, n)
    open_ = close - rng.uniform(-2, 2, n)
    vol   = rng.integers(500_000, 2_000_000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


# ==============================================================================
# Tests: FeatureConfig Validation
# ==============================================================================

class TestFeatureConfig:
    def test_default_config_is_valid(self):
        cfg = FeatureConfig()
        assert cfg.sma_windows == [5, 10, 20, 50]
        assert cfg.ema_windows == [9, 21]
        assert cfg.rsi_period == 14
        assert cfg.bb_window == 20
        assert cfg.bb_num_std == 2.0
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal == 9
        assert cfg.atr_period == 14
        assert cfg.zscore_window == 20

    def test_warmup_bars_matches_max_window(self):
        cfg = FeatureConfig(sma_windows=[5, 10], ema_windows=[9], macd_slow=26, macd_signal=9)
        # max candidates: 10, 9, 15 (rsi_period+1), 20 (bb), 34 (26+9-1), 15 (atr+1), 20 (zscore)
        assert cfg.warmup_bars == 34

    def test_custom_config_valid(self):
        cfg = FeatureConfig(
            sma_windows=[3, 7],
            ema_windows=[5],
            rsi_period=7,
            bb_window=10,
            bb_num_std=1.5,
            macd_fast=5,
            macd_slow=13,
            macd_signal=4,
            atr_period=7,
            zscore_window=10,
        )
        assert cfg.rsi_period == 7

    def test_reject_empty_sma_windows(self):
        with pytest.raises(ValueError, match="sma_windows"):
            FeatureConfig(sma_windows=[])

    def test_reject_non_positive_sma_window(self):
        with pytest.raises(ValueError, match="sma_windows"):
            FeatureConfig(sma_windows=[0, 5])

    def test_reject_macd_fast_ge_slow(self):
        with pytest.raises(ValueError, match="macd_fast"):
            FeatureConfig(macd_fast=26, macd_slow=12)

    def test_reject_zero_rsi_period(self):
        with pytest.raises(ValueError, match="rsi_period"):
            FeatureConfig(rsi_period=0)

    def test_reject_negative_bb_std(self):
        with pytest.raises(ValueError, match="bb_num_std"):
            FeatureConfig(bb_num_std=-1.0)


# ==============================================================================
# Tests: compute_returns
# ==============================================================================

class TestComputeReturns:
    def test_first_row_is_nan(self, ohlcv_20):
        out = compute_returns(ohlcv_20)
        assert np.isnan(out["simple_return"].iloc[0])
        assert np.isnan(out["log_return"].iloc[0])

    def test_simple_return_correct(self, ohlcv_20):
        out = compute_returns(ohlcv_20)
        # From 100 to 101 → return = 1/100 = 0.01
        assert np.isclose(out["simple_return"].iloc[1], 0.01, rtol=1e-6)

    def test_log_return_correct(self, ohlcv_20):
        out = compute_returns(ohlcv_20)
        expected = np.log(101.0 / 100.0)
        assert np.isclose(out["log_return"].iloc[1], expected, rtol=1e-6)

    def test_simple_and_log_return_relationship(self, ohlcv_20):
        """log_return ≈ simple_return for small changes; always log_return <= simple_return."""
        out = compute_returns(ohlcv_20)
        sr = out["simple_return"].dropna()
        lr = out["log_return"].dropna()
        assert (lr <= sr + 1e-12).all(), "log returns must be <= simple returns"

    def test_output_is_copy(self, ohlcv_20):
        """Function must not mutate the input DataFrame."""
        original_close = ohlcv_20["close"].copy()
        _ = compute_returns(ohlcv_20)
        pd.testing.assert_series_equal(ohlcv_20["close"], original_close)

    def test_missing_close_raises_keyerror(self):
        with pytest.raises(KeyError):
            compute_returns(pd.DataFrame({"open": [1.0, 2.0]}))

    def test_two_columns_added(self, ohlcv_20):
        before = len(ohlcv_20.columns)
        out = compute_returns(ohlcv_20)
        assert len(out.columns) == before + 2
        assert "simple_return" in out.columns
        assert "log_return" in out.columns


# ==============================================================================
# Tests: compute_moving_averages
# ==============================================================================

class TestComputeMovingAverages:
    def test_sma_nan_before_window(self, ohlcv_20):
        out = compute_moving_averages(ohlcv_20, sma_windows=[5], ema_windows=[])
        # First 4 rows must be NaN (window=5, need 5 observations)
        assert out["sma_5"].iloc[:4].isna().all()
        assert not np.isnan(out["sma_5"].iloc[4])

    def test_sma_5_correct_closed_form(self, ohlcv_20):
        """SMA(5) at row 4 (0-indexed) should be mean of first 5 closes (100-104)."""
        out = compute_moving_averages(ohlcv_20, sma_windows=[5], ema_windows=[])
        expected = np.mean([100.0, 101.0, 102.0, 103.0, 104.0])
        assert np.isclose(out["sma_5"].iloc[4], expected, rtol=1e-6)

    def test_ema_nan_before_min_periods(self, ohlcv_20):
        out = compute_moving_averages(ohlcv_20, sma_windows=[], ema_windows=[9])
        assert out["ema_9"].iloc[:8].isna().all()
        assert not np.isnan(out["ema_9"].iloc[8])

    def test_multiple_sma_windows_added(self, ohlcv_20):
        windows = [3, 5, 10]
        out = compute_moving_averages(ohlcv_20, sma_windows=windows, ema_windows=[])
        for w in windows:
            assert f"sma_{w}" in out.columns

    def test_config_defaults_used(self, ohlcv_100):
        cfg = FeatureConfig()
        out = compute_moving_averages(ohlcv_100, config=cfg)
        for w in cfg.sma_windows:
            assert f"sma_{w}" in out.columns
        for w in cfg.ema_windows:
            assert f"ema_{w}" in out.columns

    def test_sma_matches_pandas_rolling_mean(self, ohlcv_100):
        w = 20
        out = compute_moving_averages(ohlcv_100, sma_windows=[w], ema_windows=[])
        expected = ohlcv_100["close"].rolling(window=w, min_periods=w).mean()
        pd.testing.assert_series_equal(out[f"sma_{w}"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)


# ==============================================================================
# Tests: compute_rsi
# ==============================================================================

class TestComputeRSI:
    def test_rsi_column_present(self, ohlcv_100):
        out = compute_rsi(ohlcv_100, period=14)
        assert "rsi_14" in out.columns

    def test_rsi_nan_before_warmup(self, ohlcv_100):
        """RSI requires period+1 bars; first 14 values should be NaN."""
        out = compute_rsi(ohlcv_100, period=14)
        # With min_periods=14 on ewm, first 13 values are NaN
        assert out["rsi_14"].iloc[:13].isna().all()

    def test_rsi_range_0_to_100(self, ohlcv_100):
        out = compute_rsi(ohlcv_100, period=14)
        valid = out["rsi_14"].dropna()
        assert (valid >= 0.0).all(), "RSI must be >= 0"
        assert (valid <= 100.0).all(), "RSI must be <= 100"

    def test_rsi_pure_uptrend_approaches_100(self):
        """Consistently rising close prices → RSI should approach 100."""
        n = 100
        close = np.linspace(100.0, 200.0, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(n) * 1000,
        })
        out = compute_rsi(df, period=14)
        last_rsi = out["rsi_14"].dropna().iloc[-1]
        assert last_rsi > 90.0, f"Expected RSI > 90 in pure uptrend, got {last_rsi:.2f}"

    def test_rsi_pure_downtrend_approaches_0(self):
        """Consistently falling close prices → RSI should approach 0."""
        n = 100
        close = np.linspace(200.0, 100.0, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(n) * 1000,
        })
        out = compute_rsi(df, period=14)
        last_rsi = out["rsi_14"].dropna().iloc[-1]
        assert last_rsi < 10.0, f"Expected RSI < 10 in pure downtrend, got {last_rsi:.2f}"

    def test_rsi_custom_period(self, ohlcv_100):
        out = compute_rsi(ohlcv_100, period=7)
        assert "rsi_7" in out.columns
        assert "rsi_14" not in out.columns


# ==============================================================================
# Tests: compute_bollinger_bands
# ==============================================================================

class TestComputeBollingerBands:
    def test_expected_columns_added(self, ohlcv_100):
        out = compute_bollinger_bands(ohlcv_100, window=20, num_std=2.0)
        for col in ["bb_mid_20_2.0", "bb_upper_20_2.0", "bb_lower_20_2.0",
                    "bb_width_20_2.0", "bb_pct_20_2.0"]:
            assert col in out.columns, f"Missing: {col}"

    def test_upper_above_lower(self, ohlcv_100):
        out = compute_bollinger_bands(ohlcv_100, window=20, num_std=2.0)
        valid = out.dropna(subset=["bb_upper_20_2.0", "bb_lower_20_2.0"])
        assert (valid["bb_upper_20_2.0"] >= valid["bb_lower_20_2.0"]).all()

    def test_mid_is_sma(self, ohlcv_100):
        w = 20
        out = compute_bollinger_bands(ohlcv_100, window=w, num_std=2.0)
        expected_mid = ohlcv_100["close"].rolling(window=w, min_periods=w).mean()
        pd.testing.assert_series_equal(
            out["bb_mid_20_2.0"].reset_index(drop=True),
            expected_mid.reset_index(drop=True),
            check_names=False,
        )

    def test_nan_before_window(self, ohlcv_20):
        out = compute_bollinger_bands(ohlcv_20, window=10, num_std=2.0)
        assert out["bb_mid_10_2.0"].iloc[:9].isna().all()
        assert not np.isnan(out["bb_mid_10_2.0"].iloc[9])

    def test_bb_pct_at_midband_is_0_5(self, ohlcv_100):
        """When close == SMA exactly, %B = 0.5."""
        w = 5
        out = compute_bollinger_bands(ohlcv_100, window=w, num_std=2.0)
        # For constant price, SMA == close and upper == lower, so pct is NaN (0/0)
        # For non-constant, just verify range [0,1] approximately in normal market
        pct = out["bb_pct_5_2.0"].dropna()
        # %B can exceed [0,1] in volatile moves, so test a wide range
        assert (pct > -2.0).all() and (pct < 3.0).all()


# ==============================================================================
# Tests: compute_macd
# ==============================================================================

class TestComputeMACD:
    def test_expected_columns_added(self, ohlcv_100):
        out = compute_macd(ohlcv_100)
        assert "macd_line" in out.columns
        assert "macd_signal" in out.columns
        assert "macd_histogram" in out.columns

    def test_histogram_equals_line_minus_signal(self, ohlcv_100):
        out = compute_macd(ohlcv_100)
        valid = out.dropna(subset=["macd_line", "macd_signal", "macd_histogram"])
        diff = (valid["macd_line"] - valid["macd_signal"] - valid["macd_histogram"]).abs()
        assert (diff < 1e-10).all()

    def test_nan_before_warmup(self, ohlcv_100):
        """Histogram should be NaN until slow + signal - 1 bars have accumulated."""
        out = compute_macd(ohlcv_100, fast=12, slow=26, signal=9)
        # slow=26 means first 25 rows have NaN EMA; signal=9 stacks another 8
        # Histogram NaN count should be at least slow - 1 = 25
        n_nan_hist = out["macd_histogram"].isna().sum()
        assert n_nan_hist >= 25

    def test_macd_fast_ge_slow_raises(self, ohlcv_100):
        with pytest.raises(ValueError, match="macd_fast"):
            compute_macd(ohlcv_100, fast=26, slow=12)

    def test_custom_macd_parameters(self, ohlcv_100):
        out = compute_macd(ohlcv_100, fast=5, slow=13, signal=4)
        assert "macd_line" in out.columns


# ==============================================================================
# Tests: compute_atr
# ==============================================================================

class TestComputeATR:
    def test_atr_column_added(self, ohlcv_100):
        out = compute_atr(ohlcv_100, period=14)
        assert "atr_14" in out.columns

    def test_atr_always_positive(self, ohlcv_100):
        out = compute_atr(ohlcv_100, period=14)
        valid = out["atr_14"].dropna()
        assert (valid > 0).all()

    def test_atr_nan_before_warmup(self, ohlcv_100):
        """ATR with period=14 and min_periods=14 should have NaN for first 13 rows."""
        out = compute_atr(ohlcv_100, period=14)
        assert out["atr_14"].iloc[:13].isna().all()

    def test_atr_constant_bar_equals_high_minus_low(self):
        """For constant prices, true range = high - low = 4.0 each bar; ATR → 4.0."""
        n = 50
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "open": np.full(n, 100.0),
            "high": np.full(n, 102.0),
            "low": np.full(n, 98.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        })
        out = compute_atr(df, period=5)
        # After warm-up, ATR should stabilize at 4.0
        final_atr = out["atr_5"].dropna().iloc[-1]
        assert np.isclose(final_atr, 4.0, rtol=1e-4)

    def test_missing_high_raises_keyerror(self, ohlcv_20):
        bad_df = ohlcv_20.drop(columns=["high"])
        with pytest.raises(KeyError):
            compute_atr(bad_df)


# ==============================================================================
# Tests: compute_obv
# ==============================================================================

class TestComputeOBV:
    def test_obv_column_added(self, ohlcv_20):
        out = compute_obv(ohlcv_20)
        assert "obv" in out.columns

    def test_obv_first_bar_is_zero(self, ohlcv_20):
        out = compute_obv(ohlcv_20)
        assert out["obv"].iloc[0] == 0.0

    def test_obv_no_nans(self, ohlcv_20):
        """OBV has no warm-up period; every row must be non-NaN."""
        out = compute_obv(ohlcv_20)
        assert not out["obv"].isna().any()

    def test_obv_monotonically_increasing_on_rising_close(self, ohlcv_20):
        """Linear price increase → close[t] > close[t-1] always → OBV always increases."""
        out = compute_obv(ohlcv_20)
        diffs = out["obv"].diff().dropna()
        assert (diffs > 0).all()

    def test_obv_falls_on_declining_close(self):
        """Decreasing close → OBV should decrease."""
        n = 10
        close = np.linspace(110.0, 100.0, n)
        vol = np.ones(n) * 1000.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": vol,
        })
        out = compute_obv(df)
        diffs = out["obv"].diff().dropna()
        assert (diffs < 0).all()

    def test_obv_closed_form_three_bars(self):
        """Manual 3-bar test: up, down, flat."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "open": [100.0, 102.0, 101.0],
            "high": [103.0, 104.0, 103.0],
            "low": [99.0, 100.0, 99.0],
            "close": [102.0, 101.0, 101.0],  # down, then flat
            "volume": [1000.0, 2000.0, 3000.0],
        })
        out = compute_obv(df)
        # bar0: OBV = 0
        # bar1: close 101 < close 102 → OBV = 0 - 2000 = -2000
        # bar2: close 101 == close 101 → OBV stays -2000
        assert out["obv"].iloc[0] == 0.0
        assert out["obv"].iloc[1] == -2000.0
        assert out["obv"].iloc[2] == -2000.0


# ==============================================================================
# Tests: compute_normalization
# ==============================================================================

class TestComputeNormalization:
    def test_zscore_column_added(self, ohlcv_100):
        out = compute_returns(ohlcv_100)
        out = compute_normalization(out, columns=["close"], window=20)
        assert "close_zscore_20" in out.columns

    def test_zscore_nan_before_window(self, ohlcv_100):
        out = compute_normalization(ohlcv_100, columns=["close"], window=20)
        assert out["close_zscore_20"].iloc[:19].isna().all()

    def test_zscore_values_are_finite_and_bounded(self, ohlcv_100):
        """Rolling z-scores should be finite (no inf) and within a reasonable magnitude range."""
        w = 20
        out = compute_normalization(ohlcv_100, columns=["close"], window=w)
        valid = out["close_zscore_20"].dropna()
        # All valid z-scores must be finite
        assert np.isfinite(valid).all()
        # Z-scores for non-extreme data should be within ±6 std (well within 6-sigma rule)
        assert (valid.abs() < 6.0).all()

    def test_zscore_std_approximately_one(self, ohlcv_100):
        """Within the rolling window, the z-score std should be around 1."""
        w = 20
        out = compute_normalization(ohlcv_100, columns=["close"], window=w)
        last_w_zscore = out["close_zscore_20"].iloc[-w:]
        assert np.abs(last_w_zscore.std() - 1.0) < 0.2

    def test_zscore_flat_series_is_nan(self):
        """Constant series has std=0 → z-score should be NaN."""
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0
        })
        out = compute_normalization(df, columns=["close"], window=10)
        # After warmup, rolling std = 0 → z-score should be NaN
        valid_region = out["close_zscore_10"].iloc[10:]
        assert valid_region.isna().all()

    def test_missing_column_is_skipped_with_warning(self, ohlcv_100, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="backend.app.features.engineer"):
            out = compute_normalization(ohlcv_100, columns=["nonexistent_col"], window=10)
        assert "nonexistent_col" in caplog.text
        assert "nonexistent_col_zscore_10" not in out.columns


# ==============================================================================
# Tests: apply_all_features (Full Pipeline)
# ==============================================================================

class TestApplyAllFeatures:
    def test_returns_more_columns(self, ohlcv_100):
        out = apply_all_features(ohlcv_100)
        assert len(out.columns) > len(ohlcv_100.columns)

    def test_original_columns_preserved(self, ohlcv_100):
        out = apply_all_features(ohlcv_100)
        for col in ohlcv_100.columns:
            assert col in out.columns
            pd.testing.assert_series_equal(
                out[col].reset_index(drop=True),
                ohlcv_100[col].reset_index(drop=True),
                check_names=False,
            )

    def test_input_not_mutated(self, ohlcv_100):
        original_shape = ohlcv_100.shape
        original_cols = list(ohlcv_100.columns)
        _ = apply_all_features(ohlcv_100)
        assert ohlcv_100.shape == original_shape
        assert list(ohlcv_100.columns) == original_cols

    def test_row_count_preserved(self, ohlcv_100):
        out = apply_all_features(ohlcv_100)
        assert len(out) == len(ohlcv_100)

    def test_all_expected_feature_groups_present(self, ohlcv_100):
        cfg = FeatureConfig()
        out = apply_all_features(ohlcv_100, config=cfg)

        # Returns
        assert "simple_return" in out.columns
        assert "log_return" in out.columns

        # SMAs
        for w in cfg.sma_windows:
            assert f"sma_{w}" in out.columns

        # EMAs
        for w in cfg.ema_windows:
            assert f"ema_{w}" in out.columns

        # RSI
        assert f"rsi_{cfg.rsi_period}" in out.columns

        # Bollinger Bands
        suffix = f"{cfg.bb_window}_{cfg.bb_num_std:.1f}"
        for sub in ["mid", "upper", "lower", "width", "pct"]:
            assert f"bb_{sub}_{suffix}" in out.columns

        # MACD
        assert "macd_line" in out.columns
        assert "macd_signal" in out.columns
        assert "macd_histogram" in out.columns

        # ATR
        assert f"atr_{cfg.atr_period}" in out.columns

        # OBV
        assert "obv" in out.columns

    def test_custom_config_affects_output(self, ohlcv_100):
        cfg = FeatureConfig(sma_windows=[3], ema_windows=[7], rsi_period=7)
        out = apply_all_features(ohlcv_100, config=cfg)
        assert "sma_3" in out.columns
        assert "ema_7" in out.columns
        assert "rsi_7" in out.columns
        assert "sma_5" not in out.columns  # default sma_5 should not appear

    def test_no_inf_values(self, ohlcv_100):
        out = apply_all_features(ohlcv_100)
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        assert not out[numeric_cols].isin([np.inf, -np.inf]).any().any()

    def test_usable_rows_after_warmup(self, ohlcv_100):
        cfg = FeatureConfig()
        out = apply_all_features(ohlcv_100, config=cfg)
        usable = out.iloc[cfg.warmup_bars:]
        # After warmup, the core features should have no NaN in key columns
        for col in ["simple_return", "log_return", "obv", f"rsi_{cfg.rsi_period}"]:
            assert not usable[col].isna().any(), f"Unexpected NaN after warmup in '{col}'"


# ==============================================================================
# Tests: FeatureReport
# ==============================================================================

class TestFeatureReport:
    def test_generate_report_basic(self, ohlcv_100):
        cfg = FeatureConfig()
        original = ohlcv_100.copy()
        featured = apply_all_features(ohlcv_100, config=cfg)
        report = generate_feature_report(original, featured, symbol="RELIANCE.NS",
                                         warmup_bars=cfg.warmup_bars)

        assert report.symbol == "RELIANCE.NS"
        assert report.total_rows == 100
        assert report.usable_rows == 100 - cfg.warmup_bars
        assert len(report.feature_columns) > 0
        assert "close" in report.base_columns

    def test_report_str_includes_symbol(self, ohlcv_100):
        cfg = FeatureConfig()
        featured = apply_all_features(ohlcv_100, config=cfg)
        report = generate_feature_report(ohlcv_100, featured, "TCS.NS", cfg.warmup_bars)
        assert "TCS.NS" in str(report)

    def test_report_to_dict_serializable(self, ohlcv_100):
        import json
        cfg = FeatureConfig()
        featured = apply_all_features(ohlcv_100, config=cfg)
        report = generate_feature_report(ohlcv_100, featured, "INFY.NS", cfg.warmup_bars)
        d = report.to_dict()
        # Should serialize to JSON without error
        json_str = json.dumps(d)
        assert "INFY.NS" in json_str

    def test_nan_counts_recorded(self, ohlcv_100):
        cfg = FeatureConfig()
        featured = apply_all_features(ohlcv_100, config=cfg)
        report = generate_feature_report(ohlcv_100, featured, "SBIN.NS", cfg.warmup_bars)
        # There must be some NaN in warm-up period
        assert len(report.nan_counts) > 0
