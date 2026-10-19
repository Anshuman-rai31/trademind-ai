"""TradeMind Phase 7 Risk Models & Intervention Audit Data Structures.

Canonical enumerations and dataclasses documenting risk-management evaluations,
limit breaches, and intervention audit trails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd


class InterventionStatus(str, Enum):
    """Decision status reached by the risk manager."""
    ALLOWED = "ALLOWED"
    REJECTED = "REJECTED"
    CONSTRAINED = "CONSTRAINED"
    MODIFIED = "MODIFIED"


class InterventionReason(str, Enum):
    """Categorical explanation for a risk intervention."""
    NONE = "NONE"
    MAX_DRAWDOWN_EXCEEDED = "MAX_DRAWDOWN_EXCEEDED"
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"
    RISK_COOLDOWN_ACTIVE = "RISK_COOLDOWN_ACTIVE"
    POSITION_EXPOSURE_EXCEEDED = "POSITION_EXPOSURE_EXCEEDED"
    PORTFOLIO_EXPOSURE_EXCEEDED = "PORTFOLIO_EXPOSURE_EXCEEDED"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_PORTFOLIO_VALUE = "INVALID_PORTFOLIO_VALUE"


@dataclass
class RiskIntervention:
    """Audit log entry capturing a proposed trade evaluation by the risk manager.
    
    Attributes:
        timestamp: Time of the decision bar.
        step: Optional simulation/episode step index.
        symbol: Ticker symbol being evaluated.
        proposed_action: Action or order requested before risk checks (e.g. 'BUY').
        status: Decision status (ALLOWED, REJECTED, CONSTRAINED).
        reason: Enumerated category explaining the decision.
        details: Human-readable diagnostic rationale.
        portfolio_nav: Portfolio Net Asset Value at time of check.
        peak_nav: Running high-water mark NAV.
        current_drawdown: Peak-to-current drawdown proportion (0.0 to 1.0).
        current_daily_loss: Single-session loss percentage from previous bar.
        resulting_action: Final action or order passed to simulator (e.g. 'HOLD' if rejected).
    """
    timestamp: Any
    step: Optional[int]
    symbol: str
    proposed_action: str
    status: InterventionStatus
    reason: InterventionReason
    details: str
    portfolio_nav: float
    peak_nav: float
    current_drawdown: float
    current_daily_loss: float
    resulting_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert intervention record to serializable dictionary."""
        d = asdict(self)
        d["timestamp"] = str(self.timestamp)
        d["status"] = self.status.value
        d["reason"] = self.reason.value
        d["portfolio_nav"] = round(self.portfolio_nav, 2)
        d["portfolio_value"] = round(self.portfolio_nav, 2)
        d["peak_nav"] = round(self.peak_nav, 2)
        d["current_drawdown"] = round(self.current_drawdown, 4)
        d["current_daily_loss"] = round(self.current_daily_loss, 4)
        return d
