"""TradeMind Portfolio Simulator & Baseline Backtesting Package (Phase 4).

Provides an institutional-grade portfolio backtesting environment, realistic
execution modeling (transaction fees, slippage, cash constraints), performance
metrics calculation, and baseline benchmark strategies (Buy-and-Hold and
Moving Average Crossover).

Key Modules:
    - engine: PortfolioSimulator orchestrator with strict timing (signal at t, fill at t+1)
    - portfolio: Portfolio ledger tracking cash, holdings, cost basis, and snapshots
    - metrics: PerformanceMetrics calculation (CAGR, Sharpe, Max Drawdown, Turnover)
    - models: Order, Trade, PortfolioSnapshot, SimulationConfig, OrderSide, OrderStatus
    - strategies: BaseStrategy, BuyAndHoldStrategy, MACrossoverStrategy
    - reporting: Backtest formatting and comparative reporting tools
"""

from .engine import PortfolioSimulator, SimulationResult
from .exceptions import (
    ExecutionError,
    InsufficientCashError,
    InsufficientHoldingsError,
    InvalidOrderError,
    LookAheadBiasError,
    MetricsCalculationError,
    SimulationError,
)
from .metrics import PerformanceMetrics, calculate_metrics
from .models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    SimulationConfig,
    Trade,
)
from .portfolio import Portfolio
from .reporting import compare_strategies, export_result_json, format_backtest_summary
from .strategies import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MACrossoverStrategy,
)

__all__ = [
    # Core Engine
    "PortfolioSimulator",
    "SimulationResult",
    "Portfolio",
    # Data Models
    "Order",
    "Trade",
    "PortfolioSnapshot",
    "SimulationConfig",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    # Metrics
    "PerformanceMetrics",
    "calculate_metrics",
    # Strategies
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
    # Reporting
    "format_backtest_summary",
    "compare_strategies",
    "export_result_json",
    # Exceptions
    "SimulationError",
    "InsufficientCashError",
    "InsufficientHoldingsError",
    "InvalidOrderError",
    "ExecutionError",
    "LookAheadBiasError",
    "MetricsCalculationError",
]
