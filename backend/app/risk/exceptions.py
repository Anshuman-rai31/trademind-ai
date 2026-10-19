"""TradeMind Phase 7 Risk Management Exceptions.

Domain-specific exceptions for risk limits, exposure boundaries, and drawdown controls.
"""


class RiskError(Exception):
    """Base exception for all TradeMind risk management errors."""
    pass


class RiskLimitBreachError(RiskError):
    """Raised when an action breaches a non-recoverable risk threshold."""
    pass


class InvalidPortfolioStateError(RiskError):
    """Raised when portfolio equity or market prices are non-finite or corrupt."""
    pass
