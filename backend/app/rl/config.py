"""TradeMind Phase 5 Environment Configuration.

Configurable hyperparameters and constraints for the Gymnasium trading environment.
Ensures 100% architectural alignment with Phase 4 Portfolio Simulator settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..risk.config import RiskConfig
from ..simulation.models import SimulationConfig


@dataclass
class EnvConfig:
    """Configuration parameters for Gymnasium TradingEnvironment.
    
    Attributes:
        initial_cash: Starting portfolio capital in base currency (INR 100,000).
        fee_rate: Proportional transaction fee rate (default 0.0003 = 3 bps).
        min_fee: Fixed minimum fee per filled trade (default 0.0 INR).
        slippage_rate: Proportional execution slippage penalty (default 0.0005 = 5 bps).
        allow_fractional: Whether to allow fractional share orders (default False for NSE spot equities).
        min_trade_size: Minimum executable share quantity (default 1 share).
        allocation_pct: Proportion of unencumbered cash deployed on a BUY action
                        (default 0.95, providing a 5% buffer to guarantee fee and slippage coverage).
        reward_scaling: Multiplicative scaling factor for log portfolio returns (default 1.0).
        max_steps: Optional maximum episode duration in steps before truncation.
        warmup_bars: Minimum warm-up bars required for indicator calculations (default 50).
        feature_columns: Optional explicit list of feature column names for observation space.
        include_portfolio_state: Whether to append portfolio state features (cash ratio, position, drawdown).
        execution_price_col: Price column used for execution fills at t+1 (default 'open').
        valuation_price_col: Price column used for mark-to-market NAV valuation (default 'close').
        min_obs_bound: Lower bound for observation space box clipping.
        max_obs_bound: Upper bound for observation space box clipping.
    """
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0003
    min_fee: float = 0.0
    slippage_rate: float = 0.0005
    allow_fractional: bool = False
    min_trade_size: float = 1.0
    allocation_pct: float = 0.95
    reward_scaling: float = 1.0
    max_steps: Optional[int] = None
    warmup_bars: int = 50
    feature_columns: Optional[List[str]] = None
    include_portfolio_state: bool = True
    execution_price_col: str = "open"
    valuation_price_col: str = "close"
    min_obs_bound: float = -10.0
    max_obs_bound: float = 10.0
    risk_config: Optional[RiskConfig] = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be strictly positive, got {self.initial_cash}")
        if self.fee_rate < 0:
            raise ValueError(f"fee_rate cannot be negative, got {self.fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate cannot be negative, got {self.slippage_rate}")
        if not (0.0 < self.allocation_pct <= 1.0):
            raise ValueError(f"allocation_pct must be in (0, 1], got {self.allocation_pct}")
        if self.warmup_bars < 0:
            raise ValueError(f"warmup_bars must be non-negative, got {self.warmup_bars}")
        if self.reward_scaling <= 0:
            raise ValueError(f"reward_scaling must be strictly positive, got {self.reward_scaling}")

    def to_simulation_config(self) -> SimulationConfig:
        """Convert to Phase 4 SimulationConfig for unified ledger calculations."""
        return SimulationConfig(
            initial_cash=self.initial_cash,
            fee_rate=self.fee_rate,
            min_fee=self.min_fee,
            slippage_rate=self.slippage_rate,
            allow_fractional=self.allow_fractional,
            min_trade_size=self.min_trade_size,
            execution_price_col=self.execution_price_col,
            valuation_price_col=self.valuation_price_col,
            reject_on_insufficient=True,
        )
