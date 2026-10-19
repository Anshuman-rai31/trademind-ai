"""TradeMind Phase 6 PPO Reinforcement Learning Training & Evaluation.

Implements institutional PPO policy training using Stable-Baselines3,
validation-based model selection, artifact persistence, and fair comparative
benchmarking against Phase 4 Buy-and-Hold and Moving Average Crossover strategies.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ..data.schema import COL_TIMESTAMP
from ..data.split import chronological_split
from ..simulation.engine import PortfolioSimulator
from ..simulation.metrics import PerformanceMetrics, calculate_metrics
from ..simulation.strategies.buy_and_hold import BuyAndHoldStrategy
from ..simulation.strategies.ma_crossover import MACrossoverStrategy
from .actions import TradingAction
from .config import EnvConfig
from .environment import TradingEnvironment

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Dataclass
# ==============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters and execution configuration for SB3 PPO training.
    
    Attributes:
        seed: Random seed for PyTorch, NumPy, and environment initialization.
        total_timesteps: Total training timesteps across all rollouts.
        learning_rate: Initial learning rate for Adam optimizer.
        n_steps: Rollout buffer horizon per update (number of steps before policy gradient update).
        batch_size: Minibatch size for PPO gradient updates (must divide n_steps).
        n_epochs: Optimization epochs per rollout buffer.
        gamma: Discount factor for future rewards.
        gae_lambda: Generalized Advantage Estimation (GAE) smoothing parameter.
        clip_range: PPO surrogate objective clipping range epsilon.
        ent_coef: Entropy coefficient for exploration regularization.
        vf_coef: Value function loss coefficient in combined PPO objective.
        max_grad_norm: Maximum gradient norm clipping threshold.
        net_arch: MLP hidden layers for actor and critic networks.
        eval_freq: Timestep frequency for running validation evaluation.
        checkpoint_freq: Timestep frequency for saving periodic model checkpoints.
        device: PyTorch device ('cpu' or 'cuda').
        model_selection_metric: Metric used on validation set to select best checkpoint ('sharpe_ratio' or 'total_return').
    """
    seed: int = 42
    total_timesteps: int = 2500
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    eval_freq: int = 500
    checkpoint_freq: int = 1000
    device: str = "cpu"
    model_selection_metric: str = "sharpe_ratio"

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                f"n_steps ({self.n_steps}) must be divisible by batch_size ({self.batch_size}) "
                f"for Stable-Baselines3 PPO."
            )
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


# ==============================================================================
# Validation Evaluation Callback & Checkpoint Selection
# ==============================================================================

class ValidationCallback(BaseCallback):
    """Periodic validation callback for model selection without data leakage.
    
    Evaluates candidate checkpoints on an isolated validation environment,
    computes Phase 4 quantitative metrics, and preserves the best-performing
    model checkpoint based on the specified model selection rule.
    """

    def __init__(
        self,
        val_env: TradingEnvironment,
        eval_freq: int = 500,
        checkpoint_dir: Optional[Path] = None,
        model_selection_metric: str = "sharpe_ratio",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.val_env: TradingEnvironment = val_env
        self.eval_freq: int = eval_freq
        self.checkpoint_dir: Optional[Path] = checkpoint_dir
        self.model_selection_metric: str = model_selection_metric
        self.best_score: float = -float("inf")
        self.evaluations: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = evaluate_policy_in_env(self.model, self.val_env)
            score = metrics.get(self.model_selection_metric, -float("inf"))

            # Log validation checkpoint
            record = {
                "timestep": int(self.num_timesteps),
                "metric_used": self.model_selection_metric,
                "score": float(score),
                "total_return": float(metrics.get("total_return", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "total_trades": int(metrics.get("total_trades", 0)),
            }
            self.evaluations.append(record)

            if self.verbose > 0:
                print(
                    f"[Val @ Step {self.num_timesteps:5d}] "
                    f"Return: {record['total_return'] * 100:+.2f}% | "
                    f"Sharpe: {record['sharpe_ratio']:+.2f} | "
                    f"MDD: {record['max_drawdown'] * 100:.2f}% | "
                    f"Trades: {record['total_trades']}"
                )

            # Check if this is the best validation model so far
            if score > self.best_score:
                self.best_score = score
                if self.checkpoint_dir:
                    best_path = self.checkpoint_dir / "best_model.zip"
                    self.model.save(str(best_path))
                    if self.verbose > 0:
                        print(f" -> Saved new best model checkpoint to {best_path} (Score: {score:.4f})")

        return True


# ==============================================================================
# Policy Evaluation Engine
# ==============================================================================

def evaluate_policy_in_env(model: PPO, env: TradingEnvironment) -> Dict[str, Any]:
    """Run deterministic policy evaluation rollout through a TradingEnvironment.
    
    Args:
        model: Trained SB3 PPO model instance.
        env: TradingEnvironment instance.
        
    Returns:
        Dictionary of quantitative performance metrics from Phase 4 ledger.
    """
    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Deterministic action selection for institutional evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    # Compute institutional performance metrics from recorded portfolio snapshots
    metrics = calculate_metrics(
        snapshots=env.portfolio.snapshots,
        trades=env.portfolio.trades,
        risk_free_rate=env.config.to_simulation_config().risk_free_rate,
    )
    return metrics.to_dict()


# ==============================================================================
# Data Partitioning with Indicator Warmup Isolation
# ==============================================================================

def prepare_splits_with_warmup(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    warmup_bars: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition dataset into chronological Train, Validation, and Test slices.
    
    Data Isolation Principles:
        1. Training: Uses strictly [0 : train_end].
        2. Validation: Evaluated on [train_end : val_end]. Includes preceding
           warmup_bars from training data strictly for technical indicator rolling
           buffers, but step() decision-making starts strictly at train_end.
        3. Test: Evaluated on [val_end : n]. Includes preceding warmup_bars from
           validation data strictly for rolling buffers, but step() decision-making
           starts strictly at val_end.
        4. Zero test data is ever seen during training or validation model selection.
        
    Args:
        df: Input OHLCV DataFrame.
        train_ratio: Training proportion (default 0.70).
        val_ratio: Validation proportion (default 0.20).
        test_ratio: Test proportion (default 0.10).
        warmup_bars: Rolling window buffer length (default 50).
        
    Returns:
        Tuple of (train_df, val_with_warmup_df, test_with_warmup_df, split_info).
    """
    splits = chronological_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        embargo_bars=0,
    )

    clean_df = df.sort_values(by="timestamp" if "timestamp" in df.columns else df.columns[0]).reset_index(drop=True)
    n = len(clean_df)
    train_end = len(splits.train)
    val_end = train_end + len(splits.validation)

    # Prepare Validation slice with preceding warmup history for rolling indicators
    val_start_with_warmup = max(0, train_end - warmup_bars)
    val_df = clean_df.iloc[val_start_with_warmup:val_end].copy().reset_index(drop=True)
    val_decision_start_idx = train_end - val_start_with_warmup

    # Prepare Final Test slice with preceding warmup history
    test_start_with_warmup = max(0, val_end - warmup_bars)
    test_df = clean_df.iloc[test_start_with_warmup:].copy().reset_index(drop=True)
    test_decision_start_idx = val_end - test_start_with_warmup

    split_info = {
        "train_rows": len(splits.train),
        "val_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "total_rows": n,
        "val_decision_start_idx": val_decision_start_idx,
        "test_decision_start_idx": test_decision_start_idx,
        "warmup_bars": warmup_bars,
        "train_dates": [splits.metadata.train_start, splits.metadata.train_end],
        "val_dates": [splits.metadata.val_start, splits.metadata.val_end],
        "test_dates": [splits.metadata.test_start, splits.metadata.test_end],
    }

    return splits.train, val_df, test_df, split_info


# ==============================================================================
# Training Orchestrator
# ==============================================================================

@dataclass
class TrainingResult:
    """Artifacts and evaluation reports resulting from PPO training run."""
    model: PPO
    train_config: TrainingConfig
    env_config: EnvConfig
    best_model_path: Optional[Path]
    final_model_path: Path
    metadata_path: Path
    validation_evaluations: List[Dict[str, Any]]
    final_validation_metrics: Dict[str, Any]
    symbol: str


def train_ppo_agent(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    val_start_idx: Optional[int] = None,
    env_config: Optional[EnvConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    symbol: Optional[str] = None,
) -> TrainingResult:
    """Train a PPO reinforcement learning agent with validation model selection.
    
    Args:
        train_df: Training partition OHLCV DataFrame (strictly training dates).
        val_df: Optional validation partition DataFrame for checkpoint selection.
        val_start_idx: Start index in val_df where decision steps begin.
        env_config: Gymnasium environment hyperparameters.
        train_config: SB3 PPO training hyperparameters.
        output_dir: Directory where checkpoints and metadata are persisted.
        symbol: Ticker symbol (inferred if None).
        
    Returns:
        TrainingResult instance containing trained model and metadata audit paths.
    """
    e_cfg = env_config or EnvConfig()
    t_cfg = train_config or TrainingConfig()

    out_path = Path(output_dir or "models/checkpoints")
    out_path.mkdir(parents=True, exist_ok=True)

    # Set deterministic random seeds
    torch.manual_seed(t_cfg.seed)
    np.random.seed(t_cfg.seed)

    # 1. Instantiate Training Environment
    train_env = TradingEnvironment(train_df, config=e_cfg, symbol=symbol)
    sym = train_env.symbol

    # 2. Instantiate Validation Environment & Callback if validation data provided
    val_callback: Optional[ValidationCallback] = None
    if val_df is not None and len(val_df) > e_cfg.warmup_bars + 3:
        val_env = TradingEnvironment(
            val_df,
            config=e_cfg,
            symbol=sym,
            start_idx=val_start_idx or e_cfg.warmup_bars,
        )
        val_callback = ValidationCallback(
            val_env=val_env,
            eval_freq=t_cfg.eval_freq,
            checkpoint_dir=out_path,
            model_selection_metric=t_cfg.model_selection_metric,
            verbose=1 if t_cfg.total_timesteps > 500 else 0,
        )

    # 3. Create Stable-Baselines3 PPO Agent
    policy_kwargs = dict(
        net_arch=dict(pi=t_cfg.net_arch, vf=t_cfg.net_arch),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=t_cfg.learning_rate,
        n_steps=t_cfg.n_steps,
        batch_size=t_cfg.batch_size,
        n_epochs=t_cfg.n_epochs,
        gamma=t_cfg.gamma,
        gae_lambda=t_cfg.gae_lambda,
        clip_range=t_cfg.clip_range,
        ent_coef=t_cfg.ent_coef,
        vf_coef=t_cfg.vf_coef,
        max_grad_norm=t_cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=t_cfg.seed,
        device=t_cfg.device,
        verbose=0,
    )

    # 4. Train Model
    logger.info("Starting PPO training for %d timesteps on %s...", t_cfg.total_timesteps, sym)
    callbacks = [val_callback] if val_callback is not None else None
    model.learn(total_timesteps=t_cfg.total_timesteps, callback=callbacks)

    # 5. Save Final Checkpoint
    final_path = out_path / "final_model.zip"
    model.save(str(final_path))

    best_path = out_path / "best_model.zip"
    if not best_path.exists():
        # If no validation checkpoint improved, save final model as best
        model.save(str(best_path))

    # 6. Evaluate Final Validation Performance
    final_val_metrics: Dict[str, Any] = {}
    if val_callback is not None:
        # Load best validation model for final validation evaluation
        best_model = PPO.load(str(best_path), env=val_callback.val_env)
        final_val_metrics = evaluate_policy_in_env(best_model, val_callback.val_env)

    # 7. Persist Metadata & Configuration
    metadata = {
        "symbol": sym,
        "train_config": t_cfg.to_dict(),
        "env_config": {
            "initial_cash": e_cfg.initial_cash,
            "fee_rate": e_cfg.fee_rate,
            "slippage_rate": e_cfg.slippage_rate,
            "allocation_pct": e_cfg.allocation_pct,
            "warmup_bars": e_cfg.warmup_bars,
        },
        "validation_evaluations": val_callback.evaluations if val_callback else [],
        "best_validation_metrics": final_val_metrics,
        "best_model_path": str(best_path),
        "final_model_path": str(final_path),
    }

    meta_path = out_path / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return TrainingResult(
        model=model,
        train_config=t_cfg,
        env_config=e_cfg,
        best_model_path=best_path,
        final_model_path=final_path,
        metadata_path=meta_path,
        validation_evaluations=val_callback.evaluations if val_callback else [],
        final_validation_metrics=final_val_metrics,
        symbol=sym,
    )


# ==============================================================================
# Fair Final Test Benchmarking
# ==============================================================================

@dataclass
class TestBenchmarkReport:
    """Comparative evaluation report comparing PPO against Phase 4 benchmarks."""
    symbol: str
    ppo_metrics: Dict[str, Any]
    buy_and_hold_metrics: Dict[str, Any]
    ma_crossover_metrics: Dict[str, Any]
    outperformed_buy_and_hold: bool
    outperformed_ma_crossover: bool

    def summary(self) -> str:
        """Format clean comparison table between PPO and Phase 4 baselines."""
        ppo = self.ppo_metrics
        bh = self.buy_and_hold_metrics
        ma = self.ma_crossover_metrics

        lines = [
            "=" * 72,
            f" TradeMind Final Test Benchmark Comparison: {self.symbol}",
            "=" * 72,
            f" {'Metric':<26} {'PPO Agent':<15} {'Buy & Hold':<15} {'MA Crossover':<15}",
            "-" * 72,
            f" {'Total Return':<26} {ppo.get('total_return', 0.0)*100:>12.2f}% {bh.get('total_return', 0.0)*100:>12.2f}% {ma.get('total_return', 0.0)*100:>12.2f}%",
            f" {'Annualized (CAGR)':<26} {ppo.get('cagr', 0.0)*100:>12.2f}% {bh.get('cagr', 0.0)*100:>12.2f}% {ma.get('cagr', 0.0)*100:>12.2f}%",
            f" {'Annualized Volatility':<26} {ppo.get('annualized_volatility', 0.0)*100:>12.2f}% {bh.get('annualized_volatility', 0.0)*100:>12.2f}% {ma.get('annualized_volatility', 0.0)*100:>12.2f}%",
            f" {'Sharpe Ratio (rf=5.0%)':<26} {ppo.get('sharpe_ratio', 0.0):>13.2f} {bh.get('sharpe_ratio', 0.0):>13.2f} {ma.get('sharpe_ratio', 0.0):>13.2f}",
            f" {'Max Drawdown':<26} {ppo.get('max_drawdown', 0.0)*100:>12.2f}% {bh.get('max_drawdown', 0.0)*100:>12.2f}% {ma.get('max_drawdown', 0.0)*100:>12.2f}%",
            f" {'Total Trades':<26} {ppo.get('total_trades', 0):>13d} {bh.get('total_trades', 0):>13d} {ma.get('total_trades', 0):>13d}",
            f" {'Total Fees (INR)':<26} {ppo.get('total_fees', 0.0):>13.2f} {bh.get('total_fees', 0.0):>13.2f} {ma.get('total_fees', 0.0):>13.2f}",
            f" {'Total Slippage (INR)':<26} {ppo.get('total_slippage_cost', 0.0):>13.2f} {bh.get('total_slippage_cost', 0.0):>13.2f} {ma.get('total_slippage_cost', 0.0):>13.2f}",
            f" {'Turnover':<26} {ppo.get('turnover', 0.0):>12.2f}x {bh.get('turnover', 0.0):>12.2f}x {ma.get('turnover', 0.0):>12.2f}x",
            "=" * 72,
            f" Outperformed Buy & Hold   : {'YES' if self.outperformed_buy_and_hold else 'NO'}",
            f" Outperformed MA Crossover : {'YES' if self.outperformed_ma_crossover else 'NO'}",
            "=" * 72,
        ]
        return "\n".join(lines)


def evaluate_on_test_period(
    model: PPO,
    test_df: pd.DataFrame,
    test_start_idx: Optional[int] = None,
    env_config: Optional[EnvConfig] = None,
) -> TestBenchmarkReport:
    """Evaluate selected PPO policy and Phase 4 benchmarks on untouched test period.
    
    Guarantees strict identical testing ground:
        - Same initial capital (INR 100,000)
        - Same execution prices (t+1 Open)
        - Same brokerage fees (3 bps) and slippage (5 bps)
        - Same daily mark-to-market valuation
        
    Args:
        model: Selected PPO model.
        test_df: Final test partition with indicator warmup history.
        test_start_idx: Index where test evaluation decision steps begin.
        env_config: Environment hyperparameters.
        
    Returns:
        TestBenchmarkReport containing comparative performance analysis.
    """
    e_cfg = env_config or EnvConfig()
    sim_cfg = e_cfg.to_simulation_config()

    # 1. Evaluate PPO Agent in Gymnasium Environment
    test_env = TradingEnvironment(
        test_df,
        config=e_cfg,
        start_idx=test_start_idx or e_cfg.warmup_bars,
    )
    sym = test_env.symbol
    ppo_metrics = evaluate_policy_in_env(model, test_env)

    # 2. Extract strictly the evaluated test slice for Phase 4 simulators
    eval_slice = test_df.iloc[test_env.start_idx:].copy().reset_index(drop=True)

    # 3. Evaluate Phase 4 Buy & Hold Benchmark
    simulator = PortfolioSimulator(config=sim_cfg)
    bh_strat = BuyAndHoldStrategy(target_symbol=sym, allocation_pct=e_cfg.allocation_pct)
    bh_result = simulator.run(eval_slice, bh_strat)
    bh_metrics = bh_result.metrics.to_dict()

    # 4. Evaluate Phase 4 Moving Average Crossover Benchmark
    ma_strat = MACrossoverStrategy(
        fast_window=10,
        slow_window=30,
        target_symbol=sym,
        allocation_pct=e_cfg.allocation_pct,
    )
    ma_result = simulator.run(eval_slice, ma_strat)
    ma_metrics = ma_result.metrics.to_dict()

    # Comparison metrics (based on total return or Sharpe ratio)
    ppo_ret = ppo_metrics.get("total_return", -float("inf"))
    bh_ret = bh_metrics.get("total_return", -float("inf"))
    ma_ret = ma_metrics.get("total_return", -float("inf"))

    return TestBenchmarkReport(
        symbol=sym,
        ppo_metrics=ppo_metrics,
        buy_and_hold_metrics=bh_metrics,
        ma_crossover_metrics=ma_metrics,
        outperformed_buy_and_hold=bool(ppo_ret > bh_ret),
        outperformed_ma_crossover=bool(ppo_ret > ma_ret),
    )
