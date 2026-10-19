"""TradeMind Phase 7 Risk Configuration.

Configurable thresholds and response policies for quantitative risk management.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class RiskConfig:
    """Centralized risk management configuration.
    
    Attributes:
        enabled: Master toggle to enable or disable risk checks (default True).
        max_position_exposure: Maximum allowable fraction of portfolio NAV in a single asset (default 1.0).
        max_portfolio_exposure: Maximum total gross leverage/exposure (default 1.0, long-only spot).
        max_drawdown_limit: Maximum permissible peak-to-trough drawdown (e.g. 0.15 = 15%).
                            Breaching this triggers a risk-restricted state blocking new BUY orders.
        max_daily_loss_limit: Maximum single-day decline in portfolio NAV (e.g. 0.05 = 5%).
                              Breaching this initiates a temporary trading cooldown.
        cooldown_bars: Number of bars to block new BUY orders following a risk breach (default 5 bars).
        allow_force_liquidation: If True, severe risk breaches trigger an emergency SELL order.
                                 If False (default), the engine blocks new exposure while allowing
                                 the strategy or agent to manage existing positions.
        log_interventions: Whether to record all risk intervention events for audit trails (default True).
    """
    enabled: bool = True
    max_position_exposure: float = 1.0
    max_portfolio_exposure: float = 1.0
    max_drawdown_limit: float = 0.15
    max_daily_loss_limit: float = 0.05
    cooldown_bars: int = 5
    allow_force_liquidation: bool = False
    log_interventions: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.max_position_exposure <= 2.0):
            raise ValueError(f"max_position_exposure must be in (0, 2], got {self.max_position_exposure}")
        if not (0.0 < self.max_portfolio_exposure <= 2.0):
            raise ValueError(f"max_portfolio_exposure must be in (0, 2], got {self.max_portfolio_exposure}")
        if not (0.0 < self.max_drawdown_limit <= 1.0):
            raise ValueError(f"max_drawdown_limit must be in (0, 1], got {self.max_drawdown_limit}")
        if not (0.0 < self.max_daily_loss_limit <= 1.0):
            raise ValueError(f"max_daily_loss_limit must be in (0, 1], got {self.max_daily_loss_limit}")
        if self.cooldown_bars < 0:
            raise ValueError(f"cooldown_bars cannot be negative, got {self.cooldown_bars}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
