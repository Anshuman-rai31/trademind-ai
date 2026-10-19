"""TradeMind Phase 8 Unit & Integration Tests: Market-Regime Detection.

Deterministic, offline test suite verifying:
- RegimeConfig validation, defaults, and boundary constraints.
- Model fitting on training data only (no lookahead contamination).
- Transformation of validation and test data without refitting preprocessors or cluster centers.
- Preservation of timestamps, symbols, and raw input data immutability.
- Multi-asset time-series support without cross-stock rolling indicator leakage.
- Missing values, warm-up handling, and InsufficientDataError protections.
- Zero future-data leakage verification: perturbing future test data does not alter training model.
- Audit metadata generation and cluster summaries with descriptive historical profiles.
- Reproducibility across multiple runs with identical random seed.
- Clean single-bar inference interface for downstream decision modules.
- Model serialization and deserialization via joblib.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from backend.app.data.split import ChronologicalSplits, chronological_split
from backend.app.regime import (
    ClusterSummary,
    DataLeakageError,
    FeatureExtractionError,
    InsufficientDataError,
    MarketRegimeDetector,
    ModelNotFittedError,
    RegimeConfig,
    RegimeModelMetadata,
    RegimeOutput,
    compute_scale_free_regime_features,
)


# ==============================================================================
# Deterministic Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_ohlcv_single():
    """120-bar single-stock synthetic OHLCV DataFrame with 3 distinct market regimes."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")

    # Regime 1: Steady uptrend with low volatility (40 bars)
    r1 = np.linspace(100.0, 140.0, 40) + np.sin(np.linspace(0, 5, 40)) * 0.5

    # Regime 2: Sharp high-volatility downtrend / crash (40 bars)
    r2 = np.linspace(140.0, 80.0, 40) + np.random.RandomState(42).normal(0, 2.5, 40)

    # Regime 3: Low-volatility sideways consolidation (40 bars)
    r3 = np.linspace(80.0, 82.0, 40) + np.sin(np.linspace(0, 10, 40)) * 0.3

    close = np.concatenate([r1, r2, r3])
    open_ = np.roll(close, 1)
    open_[0] = 99.5
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = np.full(120, 15_000, dtype=np.int64)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "INFY.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def synthetic_ohlcv_multi():
    """Multi-asset synthetic OHLCV DataFrame for 2 distinct stocks."""
    dates = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")

    # Asset A: low nominal price (INR 100)
    p_a = np.linspace(100.0, 150.0, 80)
    df_a = pd.DataFrame({
        "timestamp": dates,
        "symbol": "STOCK_A",
        "open": p_a * 0.99,
        "high": p_a * 1.02,
        "low": p_a * 0.98,
        "close": p_a,
        "volume": 10_000,
    })

    # Asset B: high nominal price (INR 3,500)
    p_b = np.linspace(3500.0, 2800.0, 80)
    df_b = pd.DataFrame({
        "timestamp": dates,
        "symbol": "STOCK_B",
        "open": p_b * 1.01,
        "high": p_b * 1.03,
        "low": p_b * 0.97,
        "close": p_b,
        "volume": 5_000,
    })

    return pd.concat([df_a, df_b], axis=0).reset_index(drop=True)


# ==============================================================================
# 1. Configuration & Parameter Validation Tests
# ==============================================================================

def test_regime_config_defaults():
    """Verify default hyperparameter configuration and sensible bounds."""
    cfg = RegimeConfig()
    assert cfg.n_clusters == 3
    assert "simple_return" in cfg.feature_names
    assert "volatility_20" in cfg.feature_names
    assert "momentum_rsi" in cfg.feature_names
    assert cfg.volatility_window == 20
    assert cfg.momentum_period == 14
    assert cfg.random_state == 42
    assert cfg.scale_features is True
    assert cfg.impute_missing is True
    assert math.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0)


def test_regime_config_invalid_parameters():
    """Verify ValueError is raised on invalid configuration parameters."""
    with pytest.raises(ValueError, match="n_clusters"):
        RegimeConfig(n_clusters=1)

    with pytest.raises(ValueError, match="feature_names"):
        RegimeConfig(feature_names=[])

    with pytest.raises(ValueError, match="volatility_window"):
        RegimeConfig(volatility_window=1)

    with pytest.raises(ValueError, match="momentum_period"):
        RegimeConfig(momentum_period=0)

    with pytest.raises(ValueError, match="Split ratios"):
        RegimeConfig(train_ratio=0.80, val_ratio=0.30, test_ratio=0.10)


# ==============================================================================
# 2. Model Fitting & Leakage Prevention Tests
# ==============================================================================

def test_detector_fit_training_only(synthetic_ohlcv_single: pd.DataFrame):
    """Verify preprocessing scaler and KMeans are fitted strictly on the training partition."""
    cfg = RegimeConfig(train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, random_state=42)
    detector = MarketRegimeDetector(cfg)

    # Perform chronological split
    splits = chronological_split(synthetic_ohlcv_single, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)
    assert len(splits.train) == 84  # 70% of 120
    assert len(splits.validation) == 24
    assert len(splits.test) == 12

    # Fit detector strictly on training data
    detector.fit(splits.train)
    assert detector.is_fitted is True
    assert detector.scaler is not None
    assert detector.model is not None

    # Scaler sample count must equal valid training rows (excluding warmup), NOT full dataset
    assert detector.scaler.n_samples_seen_ < len(synthetic_ohlcv_single)
    assert detector.scaler.n_samples_seen_ <= len(splits.train)

    # Ensure cluster count matches configured value
    assert len(detector.cluster_summaries) == 3


def test_no_future_leakage_when_predicting_val_and_test(synthetic_ohlcv_single: pd.DataFrame):
    """Verify validation and test prediction does NOT alter learned training parameters."""
    cfg = RegimeConfig(random_state=42)
    detector = MarketRegimeDetector(cfg)

    splits = chronological_split(synthetic_ohlcv_single, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)
    detector.fit(splits.train)

    # Capture training parameters
    train_scaler_mean = np.copy(detector.scaler.mean_)
    train_scaler_var = np.copy(detector.scaler.var_)
    train_cluster_centers = np.copy(detector.model.cluster_centers_)

    # Predict on validation and test partitions
    val_labels = detector.predict(splits.validation)
    test_labels = detector.predict(splits.test)

    # Verify scaler and cluster centers did NOT change during prediction
    np.testing.assert_array_equal(detector.scaler.mean_, train_scaler_mean)
    np.testing.assert_array_equal(detector.scaler.var_, train_scaler_var)
    np.testing.assert_array_equal(detector.model.cluster_centers_, train_cluster_centers)

    assert len(val_labels) == len(splits.validation)
    assert len(test_labels) == len(splits.test)


def test_future_data_perturbation_leakage_test(synthetic_ohlcv_single: pd.DataFrame):
    """Strict leakage proof: Modifying future test data must NOT change training clusters or model."""
    cfg = RegimeConfig(random_state=42)

    # 1. Base run on original dataset
    splits_orig = chronological_split(synthetic_ohlcv_single, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)
    detector_orig = MarketRegimeDetector(cfg)
    detector_orig.fit(splits_orig.train)
    orig_train_labels = detector_orig.predict(splits_orig.train)

    # 2. Perturb test dataset by multiplying future close prices by 10x
    perturbed_df = synthetic_ohlcv_single.copy()
    test_start_idx = len(splits_orig.train) + len(splits_orig.validation)
    perturbed_df.loc[test_start_idx:, "close"] *= 10.0

    splits_perturbed = chronological_split(perturbed_df, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)
    detector_perturbed = MarketRegimeDetector(cfg)
    detector_perturbed.fit(splits_perturbed.train)
    perturbed_train_labels = detector_perturbed.predict(splits_perturbed.train)

    # Bit-for-bit equivalence: Training cluster centers and assignments must be 100% IDENTICAL
    np.testing.assert_array_almost_equal(
        detector_orig.model.cluster_centers_,
        detector_perturbed.model.cluster_centers_,
    )
    np.testing.assert_array_equal(orig_train_labels, perturbed_train_labels)


# ==============================================================================
# 3. Output Datasets & Immutability Tests
# ==============================================================================

def test_raw_data_preservation_and_immutability(synthetic_ohlcv_single: pd.DataFrame):
    """Verify original input DataFrame is never mutated in-place."""
    orig_cols = list(synthetic_ohlcv_single.columns)
    orig_values = synthetic_ohlcv_single["close"].to_numpy().copy()

    detector = MarketRegimeDetector(RegimeConfig(random_state=42))
    detector.fit(synthetic_ohlcv_single)

    labeled_df = detector.predict_df(synthetic_ohlcv_single)

    # Source DataFrame must be completely unchanged
    assert list(synthetic_ohlcv_single.columns) == orig_cols
    np.testing.assert_array_equal(synthetic_ohlcv_single["close"].to_numpy(), orig_values)

    # Labeled DataFrame is a separate copy with added regime columns
    assert "regime_cluster" in labeled_df.columns
    assert "regime_label" in labeled_df.columns
    assert len(labeled_df) == len(synthetic_ohlcv_single)


def test_preservation_of_timestamps_and_symbols(synthetic_ohlcv_single: pd.DataFrame):
    """Verify output preserves exact timestamps and symbol identifiers."""
    detector = MarketRegimeDetector(RegimeConfig(random_state=42))
    output = detector.fit_predict_splits(synthetic_ohlcv_single)

    labeled = output.labeled_df
    assert "timestamp" in labeled.columns
    assert "symbol" in labeled.columns

    pd.testing.assert_series_equal(labeled["timestamp"], synthetic_ohlcv_single["timestamp"])
    pd.testing.assert_series_equal(labeled["symbol"], synthetic_ohlcv_single["symbol"])


# ==============================================================================
# 4. Multi-Asset Data Handling Tests
# ==============================================================================

def test_multi_asset_regime_features(synthetic_ohlcv_multi: pd.DataFrame):
    """Verify multi-asset features are computed per-symbol without cross-asset contamination."""
    cfg = RegimeConfig(volatility_window=10)
    features_df = compute_scale_free_regime_features(synthetic_ohlcv_multi, cfg)

    assert "volatility_10" in features_df.columns
    assert "simple_return" in features_df.columns
    assert "momentum_rsi" in features_df.columns

    # Verify per-symbol returns: first row of each symbol should be NaN (warmup)
    for sym in ["STOCK_A", "STOCK_B"]:
        sym_sub = features_df[features_df["symbol"] == sym]
        assert pd.isna(sym_sub["simple_return"].iloc[0])


def test_multi_asset_detector_clustering(synthetic_ohlcv_multi: pd.DataFrame):
    """Verify MarketRegimeDetector successfully clusters pooled multi-stock dataset."""
    cfg = RegimeConfig(n_clusters=3, min_train_samples=20, random_state=42)
    detector = MarketRegimeDetector(cfg)

    output = detector.fit_predict_splits(synthetic_ohlcv_multi)
    assert output.labeled_df is not None
    assert len(output.cluster_summaries) == 3

    # Check that both symbols exist in cluster distributions
    all_sym_counts = {}
    for summary in output.cluster_summaries:
        for s, count in summary.symbol_distribution.items():
            all_sym_counts[s] = all_sym_counts.get(s, 0) + count

    assert "STOCK_A" in all_sym_counts
    assert "STOCK_B" in all_sym_counts


# ==============================================================================
# 5. Cluster Interpretation & Metadata Tests
# ==============================================================================

def test_cluster_summaries_and_interpretation(synthetic_ohlcv_single: pd.DataFrame):
    """Verify cluster summaries provide empirical returns, volatility, and momentum statistics."""
    detector = MarketRegimeDetector(RegimeConfig(n_clusters=3, random_state=42))
    detector.fit(synthetic_ohlcv_single)

    assert len(detector.cluster_summaries) == 3
    for summary in detector.cluster_summaries:
        data = summary.to_dict()
        assert "cluster_id" in data
        assert "count" in data
        assert "pct_observations" in data
        assert "mean_return" in data
        assert "median_return" in data
        assert "mean_volatility" in data
        assert "mean_momentum" in data
        assert "interpreted_label" in data
        assert "disclaimer" in data  # Historical disclaimer requirement


def test_metadata_generation(synthetic_ohlcv_single: pd.DataFrame):
    """Verify auditable provenance metadata is generated correctly."""
    detector = MarketRegimeDetector(RegimeConfig(n_clusters=3, random_state=42))
    output = detector.fit_predict_splits(synthetic_ohlcv_single)

    meta = output.metadata
    assert meta is not None
    assert meta.n_clusters == 3
    assert meta.random_state == 42
    assert meta.train_start is not None
    assert meta.train_end is not None
    assert meta.val_start is not None
    assert meta.test_start is not None
    assert len(meta.cluster_summaries) == 3


# ==============================================================================
# 6. Edge Cases & Error Handling Tests
# ==============================================================================

def test_predict_before_fit_raises():
    """Verify ModelNotFittedError is raised when predicting before fitting."""
    detector = MarketRegimeDetector()
    sample = pd.DataFrame({"close": [100.0, 102.0]})
    with pytest.raises(ModelNotFittedError):
        detector.predict(sample)


def test_insufficient_training_data_raises():
    """Verify InsufficientDataError is raised when observations are below minimum."""
    detector = MarketRegimeDetector(RegimeConfig(min_train_samples=50))
    short_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "symbol": "TCS.NS",
        "close": np.linspace(100, 110, 10),
    })
    with pytest.raises(InsufficientDataError):
        detector.fit(short_df)


def test_single_bar_prediction(synthetic_ohlcv_single: pd.DataFrame):
    """Verify zero-lookahead single bar prediction interface for later simulation phases."""
    detector = MarketRegimeDetector(RegimeConfig(random_state=42))
    detector.fit(synthetic_ohlcv_single)

    # Valid feature dictionary
    valid_bar = {
        "simple_return": 0.015,
        "volatility_20": 0.012,
        "momentum_rsi": 58.5,
    }
    cluster_id = detector.predict_single_bar(valid_bar)
    assert cluster_id in [0, 1, 2]

    # Incomplete/NaN feature returns -1
    invalid_bar = {
        "simple_return": float("nan"),
        "volatility_20": 0.012,
        "momentum_rsi": 58.5,
    }
    assert detector.predict_single_bar(invalid_bar) == -1


# ==============================================================================
# 7. Reproducibility & Persistence Tests
# ==============================================================================

def test_reproducibility_with_fixed_seed(synthetic_ohlcv_single: pd.DataFrame):
    """Verify identical cluster assignments across multiple runs with same random_state."""
    d1 = MarketRegimeDetector(RegimeConfig(random_state=123))
    out1 = d1.fit_predict_splits(synthetic_ohlcv_single)

    d2 = MarketRegimeDetector(RegimeConfig(random_state=123))
    out2 = d2.fit_predict_splits(synthetic_ohlcv_single)

    np.testing.assert_array_equal(
        out1.labeled_df["regime_cluster"].values,
        out2.labeled_df["regime_cluster"].values,
    )
    for s1, s2 in zip(out1.cluster_summaries, out2.cluster_summaries):
        assert s1.to_dict() == s2.to_dict()


def test_model_serialization_joblib(synthetic_ohlcv_single: pd.DataFrame):
    """Verify fitted detector can be saved and loaded with identical predictions."""
    detector = MarketRegimeDetector(RegimeConfig(random_state=42))
    detector.fit(synthetic_ohlcv_single)

    orig_preds = detector.predict(synthetic_ohlcv_single)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "regime_detector.joblib")
        detector.save(save_path)

        loaded_detector = MarketRegimeDetector.load(save_path)
        assert loaded_detector.is_fitted is True

        loaded_preds = loaded_detector.predict(synthetic_ohlcv_single)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
        assert len(loaded_detector.cluster_summaries) == len(detector.cluster_summaries)
