"""TradeMind Phase 5 & 6 Reinforcement Learning Package.

Provides Gymnasium-compatible trading environments and Stable-Baselines3 PPO
agent training, validation-based model selection, artifact persistence, and
comparative benchmark evaluation against Phase 4 strategies.
"""

from gymnasium.envs.registration import register

from .actions import TradingAction
from .config import EnvConfig
from .environment import TradingEnvironment
from .training import (
    TestBenchmarkReport,
    TrainingConfig,
    TrainingResult,
    ValidationCallback,
    evaluate_on_test_period,
    evaluate_policy_in_env,
    prepare_splits_with_warmup,
    train_ppo_agent,
)

register(
    id="TradeMind-v0",
    entry_point="backend.app.rl.environment:TradingEnvironment",
)

__all__ = [
    # Environment & Actions
    "TradingAction",
    "EnvConfig",
    "TradingEnvironment",
    # Training & Evaluation
    "TrainingConfig",
    "TrainingResult",
    "ValidationCallback",
    "TestBenchmarkReport",
    "train_ppo_agent",
    "evaluate_policy_in_env",
    "evaluate_on_test_period",
    "prepare_splits_with_warmup",
]
