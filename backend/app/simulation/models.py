"""TradeMind Phase 4 Simulation Data Models.

Canonical dataclasses and enumerations for order routing, trade execution,
portfolio state tracking, and simulation configuration.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import pandas as pd


class OrderSide(str, Enum):
    """Side of an order or trade."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type of order execution."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents a trading order placed by a strategy.
    
    Attributes:
        symbol: Canonical asset ticker (e.g. 'RELIANCE.NS').
        side: OrderSide.BUY or OrderSide.SELL.
        quantity: Desired number of shares (positive value).
        created_at: Timestamp when signal was generated (bar t).
        order_type: MARKET or LIMIT.
        limit_price: Optional limit price for LIMIT orders.
        order_id: Unique identifier for the order.
        status: Current lifecycle status.
        executed_at: Timestamp when filled (bar t+1).
        filled_price: Actual execution price including slippage.
        fee: Brokerage and exchange fee deducted.
        slippage: Price impact/slippage component.
        reason: Rejection reason or operational note if rejected.
    """
    symbol: str
    side: OrderSide
    quantity: float
    created_at: pd.Timestamp
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    executed_at: Optional[pd.Timestamp] = None
    filled_price: Optional[float] = None
    fee: float = 0.0
    slippage: float = 0.0
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if not isinstance(self.side, OrderSide):
            self.side = OrderSide(self.side)


@dataclass
class Trade:
    """Represents a successfully executed transaction.
    
    Attributes:
        trade_id: Unique trade identifier.
        order_id: Correlated order ID.
        symbol: Canonical ticker symbol.
        side: OrderSide.BUY or OrderSide.SELL.
        quantity: Executed share quantity.
        base_price: Market benchmark price before slippage (e.g. Open price).
        execution_price: Final fill price after applying slippage.
        slippage: Absolute per-share slippage amount applied.
        fee: Total transaction and brokerage fees paid for this trade.
        net_cost: Total cash flow impact (negative for cash outflow, positive for inflow).
        timestamp: Fill timestamp (time t+1).
        realized_pnl: Realized profit/loss for exit trades (None on initial entries).
    """
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    base_price: float
    execution_price: float
    slippage: float
    fee: float
    net_cost: float
    timestamp: pd.Timestamp
    realized_pnl: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    """Daily point-in-time portfolio mark-to-market valuation.
    
    Attributes:
        timestamp: Valuation bar timestamp.
        cash: Uninvested cash balance in base currency (INR).
        holdings: Map of symbol -> current quantity of shares held.
        holdings_value: Total mark-to-market value of open positions.
        total_value: Total portfolio net asset value (Cash + Holdings Value).
        daily_return: Percentage change in total value from previous bar.
        cumulative_return: Percentage change in total value from initial capital.
        drawdown: Peak-to-current drop proportion: (Peak - Value) / Peak.
    """
    timestamp: pd.Timestamp
    cash: float
    holdings: Dict[str, float]
    holdings_value: float
    total_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    drawdown: float = 0.0


@dataclass
class SimulationConfig:
    """Execution and configuration parameters for portfolio simulation.
    
    Attributes:
        initial_cash: Starting portfolio capital in base currency (e.g. INR 100,000).
        fee_rate: Proportional transaction fee rate (e.g. 0.0003 = 3 bps brokerage/turnover).
        min_fee: Minimum fixed fee charged per executed transaction.
        slippage_rate: Proportional slippage rate (e.g. 0.0005 = 5 bps price penalty).
        allow_fractional: Whether to allow fractional share execution (False for standard NSE equities).
        min_trade_size: Minimum executable trade size (1 share by default).
        risk_free_rate: Annualized risk-free interest rate for Sharpe ratio calculation (default 0.05 = 5%).
        execution_price_col: Column name of execution bar price (default 'open').
        valuation_price_col: Column name of mark-to-market price (default 'close').
        reject_on_insufficient: If True, reject invalid/unaffordable orders gracefully.
                                If False, raise InsufficientCashError/InsufficientHoldingsError.
    """
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0003
    min_fee: float = 0.0
    slippage_rate: float = 0.0005
    allow_fractional: bool = False
    min_trade_size: float = 1.0
    risk_free_rate: float = 0.05
    execution_price_col: str = "open"
    valuation_price_col: str = "close"
    reject_on_insufficient: bool = True

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be strictly positive, got {self.initial_cash}")
        if self.fee_rate < 0:
            raise ValueError(f"fee_rate cannot be negative, got {self.fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate cannot be negative, got {self.slippage_rate}")
        if self.min_fee < 0:
            raise ValueError(f"min_fee cannot be negative, got {self.min_fee}")
        if self.min_trade_size <= 0:
            raise ValueError(f"min_trade_size must be positive, got {self.min_trade_size}")
