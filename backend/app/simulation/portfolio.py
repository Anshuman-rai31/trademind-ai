"""TradeMind Phase 4 Portfolio Ledger and Position Manager.

Maintains cash, equity holdings, realized profit/loss, transaction audit logs,
and daily mark-to-market portfolio snapshots. Enforces strict long-only cash
constraints and prevents negative balances.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .exceptions import (
    InsufficientCashError,
    InsufficientHoldingsError,
    InvalidOrderError,
)
from .models import (
    Order,
    OrderSide,
    OrderStatus,
    PortfolioSnapshot,
    SimulationConfig,
    Trade,
)

logger = logging.getLogger(__name__)


class Portfolio:
    """Institutional portfolio ledger with strict cash and position integrity.
    
    Attributes:
        initial_cash: Starting capital.
        cash: Current unencumbered cash balance.
        holdings: Map of symbol -> current number of shares held.
        cost_basis: Map of symbol -> volume-weighted average purchase price.
        trades: Chronological list of filled trades.
        order_history: Complete chronological audit trail of all orders (filled/rejected).
        snapshots: Chronological daily mark-to-market snapshots.
        peak_value: Running maximum portfolio equity value for drawdown calculation.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be strictly positive, got {initial_cash}")
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.holdings: Dict[str, float] = {}
        self.cost_basis: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.order_history: List[Order] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.peak_value: float = float(initial_cash)

    def get_position(self, symbol: str) -> float:
        """Return the current share quantity held for a given symbol."""
        return self.holdings.get(symbol, 0.0)

    def get_holdings_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total mark-to-market value of all open stock positions.
        
        Args:
            current_prices: Map of symbol -> valuation price (e.g. Close).
            
        Returns:
            Total market value of open equity holdings.
        """
        total = 0.0
        for symbol, qty in self.holdings.items():
            if qty > 0:
                price = current_prices.get(symbol, 0.0)
                if price <= 0:
                    logger.warning(
                        "Valuation price for %s is missing or non-positive (%s). Valuing at 0.",
                        symbol,
                        price,
                    )
                    price = 0.0
                total += qty * price
        return total

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio Net Asset Value (NAV = Cash + Holdings Value)."""
        return self.cash + self.get_holdings_value(current_prices)

    @staticmethod
    def calculate_buy_cost(
        base_price: float,
        quantity: float,
        config: SimulationConfig,
    ) -> Tuple[float, float, float]:
        """Calculate execution fill price, transaction fee, and total cash outflow for a BUY.
        
        Slippage increases the purchase price.
        Transaction fee is assessed on total traded turnover.
        
        Returns:
            Tuple of (execution_price, fee, total_cash_required).
        """
        execution_price = base_price * (1.0 + config.slippage_rate)
        turnover = execution_price * quantity
        fee = max(config.min_fee, turnover * config.fee_rate)
        total_cash_required = turnover + fee
        return execution_price, fee, total_cash_required

    @staticmethod
    def calculate_sell_proceeds(
        base_price: float,
        quantity: float,
        config: SimulationConfig,
    ) -> Tuple[float, float, float]:
        """Calculate execution fill price, transaction fee, and net cash received for a SELL.
        
        Slippage decreases the realized sale price.
        Transaction fee is deducted from gross proceeds.
        
        Returns:
            Tuple of (execution_price, fee, net_cash_received).
        """
        execution_price = base_price * (1.0 - config.slippage_rate)
        turnover = execution_price * quantity
        fee = max(config.min_fee, turnover * config.fee_rate)
        net_cash_received = max(0.0, turnover - fee)
        return execution_price, fee, net_cash_received

    def execute_order(
        self,
        order: Order,
        base_price: float,
        timestamp: pd.Timestamp,
        config: SimulationConfig,
    ) -> Optional[Trade]:
        """Execute a market order against the portfolio.
        
        Enforces cash constraints, position checks, slippage, and fees.
        Updates cash, holdings, and cost basis. Appends to audit trails.
        
        Args:
            order: Order instance to fill.
            base_price: Market benchmark price (e.g. Next Bar Open).
            timestamp: Execution bar timestamp (time t+1).
            config: Simulation configuration with fees and constraints.
            
        Returns:
            Trade instance if filled, or None if rejected.
            
        Raises:
            InsufficientCashError: If cash is insufficient and reject_on_insufficient=False.
            InsufficientHoldingsError: If holdings are insufficient and reject_on_insufficient=False.
            InvalidOrderError: If order parameters are invalid.
        """
        if base_price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"Invalid base execution price: {base_price}"
            self.order_history.append(order)
            return None

        # Check fractional share constraints
        qty = order.quantity
        if not config.allow_fractional:
            if not float(qty).is_integer():
                order.status = OrderStatus.REJECTED
                order.reason = f"Fractional shares not allowed: {qty}"
                self.order_history.append(order)
                return None
            qty = float(int(qty))

        if qty < config.min_trade_size:
            order.status = OrderStatus.REJECTED
            order.reason = f"Quantity {qty} below minimum trade size {config.min_trade_size}"
            self.order_history.append(order)
            return None

        # Process BUY
        if order.side == OrderSide.BUY:
            exec_price, fee, total_cost = self.calculate_buy_cost(base_price, qty, config)
            if total_cost > self.cash:
                msg = (
                    f"Insufficient cash for BUY {qty} of {order.symbol}: "
                    f"Required INR {total_cost:.2f}, Available INR {self.cash:.2f}"
                )
                if config.reject_on_insufficient:
                    order.status = OrderStatus.REJECTED
                    order.reason = msg
                    self.order_history.append(order)
                    logger.debug(msg)
                    return None
                raise InsufficientCashError(msg)

            # Deduct cash
            self.cash -= total_cost
            current_qty = self.holdings.get(order.symbol, 0.0)
            current_basis = self.cost_basis.get(order.symbol, 0.0)

            # Update volume-weighted cost basis
            new_qty = current_qty + qty
            if new_qty > 0:
                self.cost_basis[order.symbol] = (
                    (current_qty * current_basis) + (qty * exec_price)
                ) / new_qty
            self.holdings[order.symbol] = new_qty

            trade = Trade(
                trade_id=str(uuid.uuid4())[:8],
                order_id=order.order_id,
                symbol=order.symbol,
                side=OrderSide.BUY,
                quantity=qty,
                base_price=base_price,
                execution_price=exec_price,
                slippage=abs(exec_price - base_price),
                fee=fee,
                net_cost=-total_cost,
                timestamp=timestamp,
                realized_pnl=None,
            )

            order.status = OrderStatus.FILLED
            order.executed_at = timestamp
            order.filled_price = exec_price
            order.fee = fee
            order.slippage = abs(exec_price - base_price)

            self.trades.append(trade)
            self.order_history.append(order)
            return trade

        # Process SELL
        elif order.side == OrderSide.SELL:
            current_qty = self.holdings.get(order.symbol, 0.0)
            if qty > current_qty:
                msg = (
                    f"Insufficient holdings for SELL {qty} of {order.symbol}: "
                    f"Currently held {current_qty}"
                )
                if config.reject_on_insufficient:
                    order.status = OrderStatus.REJECTED
                    order.reason = msg
                    self.order_history.append(order)
                    logger.debug(msg)
                    return None
                raise InsufficientHoldingsError(msg)

            exec_price, fee, net_proceeds = self.calculate_sell_proceeds(base_price, qty, config)
            basis = self.cost_basis.get(order.symbol, 0.0)
            # Realized PnL = (exec_price - basis) * qty - fee
            realized_pnl = ((exec_price - basis) * qty) - fee

            self.cash += net_proceeds
            remaining_qty = current_qty - qty
            if remaining_qty <= 1e-9:
                self.holdings[order.symbol] = 0.0
                self.cost_basis[order.symbol] = 0.0
            else:
                self.holdings[order.symbol] = remaining_qty

            trade = Trade(
                trade_id=str(uuid.uuid4())[:8],
                order_id=order.order_id,
                symbol=order.symbol,
                side=OrderSide.SELL,
                quantity=qty,
                base_price=base_price,
                execution_price=exec_price,
                slippage=abs(exec_price - base_price),
                fee=fee,
                net_cost=net_proceeds,
                timestamp=timestamp,
                realized_pnl=realized_pnl,
            )

            order.status = OrderStatus.FILLED
            order.executed_at = timestamp
            order.filled_price = exec_price
            order.fee = fee
            order.slippage = abs(exec_price - base_price)

            self.trades.append(trade)
            self.order_history.append(order)
            return trade

        else:
            raise InvalidOrderError(f"Unsupported order side: {order.side}")

    def record_snapshot(
        self,
        timestamp: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> PortfolioSnapshot:
        """Compute point-in-time valuation and append daily snapshot.
        
        Args:
            timestamp: Bar timestamp (e.g. market close of day t).
            current_prices: Current mark-to-market prices.
            
        Returns:
            Appended PortfolioSnapshot.
        """
        holdings_value = self.get_holdings_value(current_prices)
        total_value = self.cash + holdings_value

        # Track running peak equity
        if total_value > self.peak_value:
            self.peak_value = total_value

        # Calculate drawdown
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = max(0.0, (self.peak_value - total_value) / self.peak_value)

        # Calculate returns
        daily_return = 0.0
        if self.snapshots:
            prev_value = self.snapshots[-1].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value

        cumulative_return = 0.0
        if self.initial_cash > 0:
            cumulative_return = (total_value - self.initial_cash) / self.initial_cash

        # Snapshot open holdings (copy dict to ensure immutability of snapshot)
        holdings_copy = {k: v for k, v in self.holdings.items() if v > 0}

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            holdings=holdings_copy,
            holdings_value=holdings_value,
            total_value=total_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
        )
        self.snapshots.append(snapshot)
        return snapshot
