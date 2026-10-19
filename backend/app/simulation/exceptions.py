"""TradeMind Phase 4 Simulation Exceptions.

Hierarchy of domain-specific actionable exceptions for portfolio simulation,
order processing, position management, and performance calculation.
"""


class SimulationError(Exception):
    """Base exception for all simulation and backtesting errors."""
    pass


class InsufficientCashError(SimulationError):
    """Raised when an order requires more cash than currently available."""
    pass


class InsufficientHoldingsError(SimulationError):
    """Raised when attempting to sell more shares than currently held."""
    pass


class InvalidOrderError(SimulationError):
    """Raised when an order specification is invalid (e.g., negative quantity, unknown symbol)."""
    pass


class ExecutionError(SimulationError):
    """Raised when order execution fails due to missing prices or execution constraints."""
    pass


class LookAheadBiasError(SimulationError):
    """Raised when an order or signal references future timestamps or future prices."""
    pass


class MetricsCalculationError(SimulationError):
    """Raised when performance metric calculations fail due to corrupt or invalid equity data."""
    pass
