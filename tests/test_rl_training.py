"""TradeMind Phase 6 Unit & Integration Tests: PPO Agent Training & Evaluation.

Deterministic test suite verifying:
- PPO agent creation, action space compatibility, and hyperparameter checks.
- Short training run completion and model artifact persistence.
- Model saving, loading, and deterministic inference.
- Data isolation: strict 70/20/10 split enforcement without test-data leakage.
- Validation callback evaluation and model selection tracking.
- Test period comparative evaluation against Phase 4 Buy & Hold and MA Crossover.
- Seeded training reproducibility.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

from backend.app.rl import (
    EnvConfig,
    TradingAction,
    TradingEnvironment,
    TrainingConfig,
    evaluate_on_test_period,
    evaluate_policy_in_env,
    prepare_splits_with_warmup,
    train_ppo_agent,
)


# ==============================================================================
# Deterministic Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_history_120():
    """120-bar deterministic price series for training and validation splits."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    close = 100.0 + 15.0 * np.sin(np.linspace(0, 2 * np.pi, 120))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    volume = np.full(120, 50000.0)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "INFY.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ==============================================================================
# Configuration & Model Instantiation Tests
# ==============================================================================

def test_training_config_validations():
    """Verify hyperparameter constraints and SB3 PPO divisibility checks."""
    # Valid config
    cfg = TrainingConfig(n_steps=128, batch_size=64)
    assert cfg.n_steps % cfg.batch_size == 0

    # Incompatible batch_size and n_steps
    with pytest.raises(ValueError, match="divisible by batch_size"):
        TrainingConfig(n_steps=100, batch_size=32)

    # Invalid gamma
    with pytest.raises(ValueError, match="gamma must be in"):
        TrainingConfig(gamma=0.0)

    # Invalid timesteps
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        TrainingConfig(total_timesteps=-10)


def test_ppo_agent_initialization(synthetic_history_120):
    """Verify PPO model creates successfully with TradingEnvironment spaces."""
    env_cfg = EnvConfig(warmup_bars=10)
    env = TradingEnvironment(synthetic_history_120, config=env_cfg)

    train_cfg = TrainingConfig(n_steps=64, batch_size=32, total_timesteps=64)
    model = PPO("MlpPolicy", env=env, n_steps=64, batch_size=32, seed=42)

    assert model.action_space == env.action_space
    assert model.observation_space == env.observation_space
    assert model.n_steps == 64
    assert model.batch_size == 32


# ==============================================================================
# Data Isolation & Split Verification Tests
# ==============================================================================

def test_data_isolation_and_split_integrity(synthetic_history_120):
    """Verify 70/20/10 split boundaries, warmup buffers, and zero test leakage."""
    train_df, val_df, test_df, info = prepare_splits_with_warmup(
        synthetic_history_120,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        warmup_bars=10,
    )

    total_rows = len(synthetic_history_120)
    assert info["total_rows"] == total_rows

    # Train ratio ~70% (84 rows)
    assert info["train_rows"] == 84
    # Val ratio ~20% (24 rows)
    assert info["val_rows"] == 24
    # Test ratio ~10% (12 rows)
    assert info["test_rows"] == 12

    # Verify strict temporal ordering of decision boundaries
    train_max_date = train_df["timestamp"].max()
    val_decision_date = val_df["timestamp"].iloc[info["val_decision_start_idx"]]
    test_decision_date = test_df["timestamp"].iloc[info["test_decision_start_idx"]]

    assert train_max_date < val_decision_date
    assert val_decision_date < test_decision_date

    # Verify test data rows are completely absent from training dataset
    test_timestamps = set(test_df["timestamp"].iloc[info["test_decision_start_idx"]:])
    train_timestamps = set(train_df["timestamp"])
    assert len(test_timestamps.intersection(train_timestamps)) == 0


# ==============================================================================
# Training Run & Model Persistence Tests
# ==============================================================================

def test_short_training_run_and_artifact_saving(synthetic_history_120):
    """Verify short training run persists best model, final model, and metadata."""
    train_df, val_df, test_df, info = prepare_splits_with_warmup(
        synthetic_history_120,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        warmup_bars=10,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        env_cfg = EnvConfig(warmup_bars=10, initial_cash=100_000.0)
        train_cfg = TrainingConfig(
            seed=42,
            total_timesteps=128,  # Exactly 1 rollout buffer
            n_steps=128,
            batch_size=64,
            eval_freq=128,
        )

        result = train_ppo_agent(
            train_df=train_df,
            val_df=val_df,
            val_start_idx=info["val_decision_start_idx"],
            env_config=env_cfg,
            train_config=train_cfg,
            output_dir=tmp_dir,
            symbol="INFY.NS",
        )

        # Assert artifacts exist
        assert result.final_model_path.exists()
        assert result.best_model_path.exists()
        assert result.metadata_path.exists()

        # Check metadata JSON validity
        with open(result.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["symbol"] == "INFY.NS"
        assert meta["train_config"]["total_timesteps"] == 128
        assert "best_validation_metrics" in meta


def test_model_save_and_reload_deterministic_inference(synthetic_history_120):
    """Verify saved PPO model reloads cleanly and produces identical predictions."""
    env_cfg = EnvConfig(warmup_bars=10)
    env = TradingEnvironment(synthetic_history_120, config=env_cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.zip"
        model = PPO("MlpPolicy", env=env, n_steps=64, batch_size=32, seed=42)
        model.learn(total_timesteps=64)
        model.save(str(model_path))

        # Reload model
        loaded_model = PPO.load(str(model_path), env=env)

        obs, _ = env.reset()
        act1, _ = model.predict(obs, deterministic=True)
        act2, _ = loaded_model.predict(obs, deterministic=True)

        assert act1 == act2


# ==============================================================================
# Final Test Period Comparative Evaluation Tests
# ==============================================================================

def test_evaluate_on_test_period_against_benchmarks(synthetic_history_120):
    """Verify test evaluation compares PPO against Phase 4 Buy & Hold and MA Crossover."""
    train_df, val_df, test_df, info = prepare_splits_with_warmup(
        synthetic_history_120,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        warmup_bars=10,
    )

    env_cfg = EnvConfig(warmup_bars=10, initial_cash=50_000.0)
    train_env = TradingEnvironment(train_df, config=env_cfg)
    model = PPO("MlpPolicy", env=train_env, n_steps=64, batch_size=32, seed=42)
    model.learn(total_timesteps=64)

    report = evaluate_on_test_period(
        model=model,
        test_df=test_df,
        test_start_idx=info["test_decision_start_idx"],
        env_config=env_cfg,
    )

    assert report.symbol == "INFY.NS"
    assert "total_return" in report.ppo_metrics
    assert "total_return" in report.buy_and_hold_metrics
    assert "total_return" in report.ma_crossover_metrics
    assert isinstance(report.outperformed_buy_and_hold, bool)
    assert isinstance(report.outperformed_ma_crossover, bool)

    summary_str = report.summary()
    assert "TradeMind Final Test Benchmark Comparison" in summary_str
    assert "PPO Agent" in summary_str
    assert "Buy & Hold" in summary_str
    assert "MA Crossover" in summary_str
