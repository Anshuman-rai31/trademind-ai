"""TradeMind Phase 6 PPO Training & Evaluation CLI.

Command-line entry point for training Stable-Baselines3 PPO agents on
historical market data and benchmarking against Phase 4 strategies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import EnvConfig
from .training import (
    TrainingConfig,
    evaluate_on_test_period,
    prepare_splits_with_warmup,
    train_ppo_agent,
)


def create_synthetic_market_data(symbol: str = "RELIANCE.NS", bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic multi-year synthetic OHLCV data for testing/offline research."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=bars, freq="D", tz="UTC")

    # Drift + cycle + noise
    t = np.linspace(0, 4 * np.pi, bars)
    trend = np.linspace(100.0, 180.0, bars)
    cycle = 15.0 * np.sin(t)
    noise = np.cumsum(rng.normal(0, 0.5, bars))
    close = trend + cycle + noise
    close = np.maximum(close, 20.0)

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, bars)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, bars)
    volume = rng.integers(20000, 100000, bars).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def main(args: Optional[list] = None) -> int:
    """CLI entry point for Phase 6 RL agent training."""
    parser = argparse.ArgumentParser(
        description="TradeMind Phase 6: PPO Agent Training & Comparative Benchmark Evaluation",
    )
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol to train on")
    parser.add_argument("--timesteps", type=int, default=2500, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout buffer length")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--eval-freq", type=int, default=500, help="Validation evaluation frequency")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 256-step verification smoke test")
    parser.add_argument("--data-file", type=str, default=None, help="Path to processed CSV dataset")
    parser.add_argument("--skip-test-eval", action="store_true", help="Skip final test period evaluation")

    parsed = parser.parse_args(args)

    print("=" * 72)
    print(" TradeMind Phase 6: PPO Reinforcement Learning Training")
    print("=" * 72)

    # 1. Load or Generate Dataset
    if parsed.data_file and Path(parsed.data_file).exists():
        print(f"[*] Loading processed dataset from: {parsed.data_file}")
        df = pd.read_csv(parsed.data_file)
    else:
        print(f"[*] Using deterministic synthetic research dataset for {parsed.symbol} (500 bars, seed={parsed.seed})")
        df = create_synthetic_market_data(symbol=parsed.symbol, bars=500, seed=parsed.seed)

    # Smoke test settings
    timesteps = 256 if parsed.smoke_test else parsed.timesteps
    eval_freq = 128 if parsed.smoke_test else parsed.eval_freq

    # 2. Partition Chronologically with Indicator Warmup Buffer
    print("[*] Partitioning dataset into 70% Train, 20% Validation, 10% Test...")
    train_df, val_df, test_df, split_info = prepare_splits_with_warmup(
        df,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        warmup_bars=50,
    )
    print(
        f"    Train: {split_info['train_rows']} bars | "
        f"Val: {split_info['val_rows']} bars | "
        f"Test: {split_info['test_rows']} bars"
    )

    env_config = EnvConfig(warmup_bars=50, initial_cash=100_000.0)
    train_config = TrainingConfig(
        seed=parsed.seed,
        total_timesteps=timesteps,
        learning_rate=parsed.lr,
        n_steps=parsed.n_steps,
        batch_size=parsed.batch_size,
        eval_freq=eval_freq,
    )

    # 3. Train PPO Agent
    print(f"[*] Launching PPO training ({train_config.total_timesteps} steps, seed={train_config.seed})...")
    result = train_ppo_agent(
        train_df=train_df,
        val_df=val_df,
        val_start_idx=split_info["val_decision_start_idx"],
        env_config=env_config,
        train_config=train_config,
        output_dir=parsed.output_dir,
        symbol=parsed.symbol,
    )
    print(f"[+] Training complete. Best model checkpoint saved to: {result.best_model_path}")

    # 4. Final Untouched Test Evaluation
    if not parsed.skip_test_eval:
        print("[*] Evaluating selected best PPO policy against Phase 4 benchmarks on final test period...")
        best_model = result.model
        from stable_baselines3 import PPO
        if result.best_model_path and Path(result.best_model_path).exists():
            best_model = PPO.load(str(result.best_model_path))

        report = evaluate_on_test_period(
            model=best_model,
            test_df=test_df,
            test_start_idx=split_info["test_decision_start_idx"],
            env_config=env_config,
        )
        print("\n" + report.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
