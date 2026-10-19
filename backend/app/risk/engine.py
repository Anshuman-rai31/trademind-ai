"""TradeMind Phase 7 Risk Management Engine.

Deterministic pre-trade risk evaluation, position sizing constraints, drawdown
threshold monitoring, daily loss limits, and intervention audit recording.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..simulation.models import Order, OrderSide, SimulationConfig
from ..simulation.portfolio import Portfolio
from .config import RiskConfig
from .models import InterventionReason, InterventionStatus, RiskIntervention

logger = logging.getLogger(__name__)


class RiskManager:
    """Institutional deterministic risk evaluation and intervention engine.
    
    Acts as a protective gateway sitting strictly between strategy/agent signal
    generation and order execution in the portfolio ledger.
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """Initialize RiskManager with configurable policy parameters.
        
        Args:
            config: Optional RiskConfig parameters.
        """
        self.config: RiskConfig = config or RiskConfig()
        self.peak_nav: float = 0.0
        self.prev_nav: float = 0.0
        self.current_drawdown: float = 0.0
        self.current_daily_loss: float = 0.0
        self.cooldown_remaining: int = 0
        self.is_restricted: bool = False
        self.interventions: List[RiskIntervention] = []

    def reset(self, initial_cash: float) -> None:
        """Reset internal risk state ledger for a new backtest or episode run."""
        self.peak_nav = float(initial_cash)
        self.prev_nav = float(initial_cash)
        self.current_drawdown = 0.0
        self.current_daily_loss = 0.0
        self.cooldown_remaining = 0
        self.is_restricted = False
        self.interventions = []

    def update_portfolio_valuation(
        self,
        timestamp: Any,
        current_nav: float,
        step: Optional[int] = None,
    ) -> None:
        """Update point-in-time drawdown, daily loss metrics, and cooldown counters.
        
        Args:
            timestamp: Valuation bar timestamp.
            current_nav: Current marked-to-market portfolio Net Asset Value.
            step: Optional episode step index.
        """
        if not self.config.enabled:
            return

        # 1. Guard against non-finite or corrupt portfolio valuations
        if np.isnan(current_nav) or np.isinf(current_nav) or current_nav <= 0.0:
            self.is_restricted = True
            logger.error(
                "Invalid portfolio valuation encountered: %s. Entering restricted state.",
                current_nav,
            )
            self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol="PORTFOLIO",
                proposed_action="VALUATION",
                status=InterventionStatus.MODIFIED,
                reason=InterventionReason.INVALID_PORTFOLIO_VALUE,
                details=f"Non-finite or non-positive NAV: {current_nav}",
                resulting_action="LOCKOUT",
            )
            return

        # Initialize tracking on first valuation
        if self.peak_nav <= 0.0:
            self.peak_nav = current_nav
            self.prev_nav = current_nav
            return

        # 2. Update High-Water Mark and Drawdown
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

        if self.peak_nav > 0:
            self.current_drawdown = max(0.0, (self.peak_nav - current_nav) / self.peak_nav)
        else:
            self.current_drawdown = 0.0

        # 3. Calculate Daily Session Loss
        if self.prev_nav > 0:
            self.current_daily_loss = max(0.0, (self.prev_nav - current_nav) / self.prev_nav)
        else:
            self.current_daily_loss = 0.0

        self.prev_nav = current_nav

        # 4. Check Risk Threshold Breaches
        drawdown_breach = self.current_drawdown > self.config.max_drawdown_limit
        daily_loss_breach = self.current_daily_loss > self.config.max_daily_loss_limit

        if drawdown_breach or daily_loss_breach:
            self.is_restricted = True
            self.cooldown_remaining = self.config.cooldown_bars
            reason = (
                InterventionReason.MAX_DRAWDOWN_EXCEEDED
                if drawdown_breach
                else InterventionReason.DAILY_LOSS_EXCEEDED
            )
            details = (
                f"Risk breach detected: Drawdown={self.current_drawdown*100:.2f}% "
                f"(limit={self.config.max_drawdown_limit*100:.1f}%), "
                f"DailyLoss={self.current_daily_loss*100:.2f}% "
                f"(limit={self.config.max_daily_loss_limit*100:.1f}%). "
                f"Cooldown engaged for {self.config.cooldown_bars} bars."
            )
            self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol="PORTFOLIO",
                proposed_action="STATE_CHECK",
                status=InterventionStatus.MODIFIED,
                reason=reason,
                details=details,
                resulting_action="RESTRICTED",
            )

        else:
            # Clean cooldown decay if no active breach
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                if self.cooldown_remaining == 0 and not drawdown_breach:
                    self.is_restricted = False
                    logger.debug("[%s] Risk cooldown released. Normal trading resumed.", timestamp)

    def evaluate_proposed_order(
        self,
        order: Order,
        portfolio: Portfolio,
        execution_price: float,
        timestamp: Any,
        step: Optional[int] = None,
    ) -> Tuple[Optional[Order], Optional[RiskIntervention]]:
        """Pre-trade risk inspection of a proposed Order before ledger execution.
        
        Args:
            order: Proposed Order instance.
            portfolio: Active Portfolio ledger.
            execution_price: Expected benchmark execution price (e.g. Open price).
            timestamp: Order evaluation timestamp.
            step: Optional episode step index.
            
        Returns:
            Tuple of (approved_order, risk_intervention). If order is rejected,
            approved_order is None. If constrained, approved_order has adjusted quantity.
        """
        if not self.config.enabled:
            return order, None

        # Sells reduce risk and are always allowed to de-risk the portfolio
        if order.side == OrderSide.SELL:
            return order, None

        # 1. Price Validity Check
        if execution_price <= 0.0 or np.isnan(execution_price) or np.isinf(execution_price):
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.INVALID_PRICE,
                details=f"Invalid execution price: {execution_price}",
                resulting_action="REJECT",
            )
            return None, intervention

        # 2. Risk-Restricted / Cooldown State Gate
        if self.is_restricted or self.cooldown_remaining > 0:
            reason = (
                InterventionReason.MAX_DRAWDOWN_EXCEEDED
                if self.current_drawdown > self.config.max_drawdown_limit
                else (
                    InterventionReason.DAILY_LOSS_EXCEEDED
                    if self.current_daily_loss > self.config.max_daily_loss_limit
                    else InterventionReason.RISK_COOLDOWN_ACTIVE
                )
            )
            details = (
                f"Blocked exposure increase: Drawdown={self.current_drawdown*100:.2f}% "
                f"(limit={self.config.max_drawdown_limit*100:.1f}%), "
                f"CooldownRemaining={self.cooldown_remaining} bars."
            )
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=reason,
                details=details,
                resulting_action="REJECT",
            )
            return None, intervention

        # 3. Cash Affordability Check (Zero phantom orders & impossible fills)
        if portfolio.cash < (order.quantity * execution_price):
            details = (
                f"Proposed order cost INR {order.quantity * execution_price:,.2f} "
                f"exceeds available cash INR {portfolio.cash:,.2f}."
            )
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.INSUFFICIENT_CASH,
                details=details,
                resulting_action="REJECT",
            )
            return None, intervention

        # 4. Portfolio-Wide Gross Exposure Limit Enforcement
        nav = portfolio.get_total_value({order.symbol: execution_price})
        if nav <= 0.0:
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.INVALID_PORTFOLIO_VALUE,
                details=f"Portfolio NAV is non-positive: {nav}",
                resulting_action="REJECT",
            )
            return None, intervention

        total_current_holdings_val = sum(
            qty * (execution_price if sym == order.symbol else portfolio.cost_basis.get(sym, execution_price))
            for sym, qty in portfolio.holdings.items()
        )
        max_portfolio_val = nav * self.config.max_portfolio_exposure
        if total_current_holdings_val >= max_portfolio_val:
            details = (
                f"Portfolio exposure limit reached: Current total exposure INR {total_current_holdings_val:,.2f} "
                f"is at max allowable INR {max_portfolio_val:,.2f} ({self.config.max_portfolio_exposure*100:.1f}% NAV)"
            )
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.PORTFOLIO_EXPOSURE_EXCEEDED,
                details=details,
                resulting_action="REJECT",
            )
            return None, intervention

        available_portfolio_val = max(0.0, max_portfolio_val - total_current_holdings_val)
        max_portfolio_shares = float(int(available_portfolio_val / execution_price))
        if order.quantity > max_portfolio_shares:
            if max_portfolio_shares < 1.0:
                details = (
                    f"Order would exceed portfolio exposure limit {self.config.max_portfolio_exposure*100:.1f}% NAV. "
                    f"Affordable headroom ({max_portfolio_shares} shares) below 1 share."
                )
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=order.symbol,
                    proposed_action=f"BUY {order.quantity}",
                    status=InterventionStatus.REJECTED,
                    reason=InterventionReason.PORTFOLIO_EXPOSURE_EXCEEDED,
                    details=details,
                    resulting_action="REJECT",
                )
                return None, intervention
            else:
                constrained_order = Order(
                    symbol=order.symbol,
                    side=OrderSide.BUY,
                    quantity=max_portfolio_shares,
                    created_at=order.created_at,
                    order_type=order.order_type,
                )
                details = (
                    f"Constrained BUY quantity from {order.quantity} to {max_portfolio_shares} "
                    f"to respect {self.config.max_portfolio_exposure*100:.1f}% portfolio limit."
                )
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=order.symbol,
                    proposed_action=f"BUY {order.quantity}",
                    status=InterventionStatus.CONSTRAINED,
                    reason=InterventionReason.PORTFOLIO_EXPOSURE_EXCEEDED,
                    details=details,
                    resulting_action=f"BUY {max_portfolio_shares}",
                )
                order = constrained_order

        # 4. Position Exposure Limit Enforcement

        current_qty = portfolio.get_position(order.symbol)
        max_position_val = nav * self.config.max_position_exposure
        current_position_val = current_qty * execution_price

        # Check if already at or exceeding limit
        if current_position_val >= max_position_val:
            details = (
                f"Position limit reached: Current position INR {current_position_val:,.2f} "
                f"is at max allowable INR {max_position_val:,.2f} ({self.config.max_position_exposure*100:.1f}% NAV)"
            )
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=order.symbol,
                proposed_action=f"BUY {order.quantity}",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.POSITION_EXPOSURE_EXCEEDED,
                details=details,
                resulting_action="REJECT",
            )
            return None, intervention

        # Calculate maximum allowable incremental shares
        available_val = max(0.0, max_position_val - current_position_val)
        max_shares = available_val / execution_price
        max_shares = float(int(max_shares))

        if order.quantity > max_shares:
            if max_shares < 1.0:
                details = (
                    f"Order would exceed position exposure limit {self.config.max_position_exposure*100:.1f}% NAV. "
                    f"Affordable headroom ({max_shares} shares) below 1 share."
                )
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=order.symbol,
                    proposed_action=f"BUY {order.quantity}",
                    status=InterventionStatus.REJECTED,
                    reason=InterventionReason.POSITION_EXPOSURE_EXCEEDED,
                    details=details,
                    resulting_action="REJECT",
                )
                return None, intervention
            else:
                # Constrain order quantity down to allowable boundary
                constrained_order = Order(
                    symbol=order.symbol,
                    side=OrderSide.BUY,
                    quantity=max_shares,
                    created_at=order.created_at,
                    order_type=order.order_type,
                )
                details = (
                    f"Constrained BUY quantity from {order.quantity} to {max_shares} "
                    f"to respect {self.config.max_position_exposure*100:.1f}% position limit."
                )
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=order.symbol,
                    proposed_action=f"BUY {order.quantity}",
                    status=InterventionStatus.CONSTRAINED,
                    reason=InterventionReason.POSITION_EXPOSURE_EXCEEDED,
                    details=details,
                    resulting_action=f"BUY {max_shares}",
                )
                return constrained_order, intervention

        # All checks passed cleanly
        return order, None

    def evaluate_proposed_action(
        self,
        action: int,
        symbol: str,
        portfolio: Portfolio,
        current_price: float,
        next_open: float,
        step: Optional[int] = None,
        timestamp: Any = None,
    ) -> Tuple[int, Optional[RiskIntervention]]:
        """Evaluate discrete Gymnasium action (0=HOLD, 1=BUY, 2=SELL) before environment execution.
        
        Args:
            action: Proposed discrete action index.
            symbol: Ticker symbol.
            portfolio: Active Portfolio ledger.
            current_price: Latest close price at decision time t.
            next_open: Expected execution open price at t+1.
            step: Optional episode step index.
            timestamp: Evaluation timestamp.
            
        Returns:
            Tuple of (effective_action, risk_intervention).
        """
        if not self.config.enabled:
            return action, None

        # Action 2 (SELL) reduces risk: permitted
        if action == 2:
            return 2, None

        # Action 0 (HOLD): permitted unless emergency force liquidation is triggered
        if action == 0:
            if (
                self.config.allow_force_liquidation
                and self.is_restricted
                and portfolio.get_position(symbol) > 0
            ):
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=symbol,
                    proposed_action="HOLD",
                    status=InterventionStatus.MODIFIED,
                    reason=InterventionReason.MAX_DRAWDOWN_EXCEEDED,
                    details="Emergency force liquidation triggered due to severe risk breach.",
                    resulting_action="SELL",
                )
                return 2, intervention
            return 0, None

        # Action 1 (BUY): Exposure-increasing action
        # 1. Price validity check
        if next_open <= 0.0 or np.isnan(next_open) or np.isinf(next_open):
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=symbol,
                proposed_action="BUY",
                status=InterventionStatus.REJECTED,
                reason=InterventionReason.INVALID_PRICE,
                details=f"Invalid execution price: {next_open}",
                resulting_action="HOLD",
            )
            return 0, intervention

        # 2. Risk-restricted or cooldown check
        if self.is_restricted or self.cooldown_remaining > 0:
            reason = (
                InterventionReason.MAX_DRAWDOWN_EXCEEDED
                if self.current_drawdown > self.config.max_drawdown_limit
                else (
                    InterventionReason.DAILY_LOSS_EXCEEDED
                    if self.current_daily_loss > self.config.max_daily_loss_limit
                    else InterventionReason.RISK_COOLDOWN_ACTIVE
                )
            )
            details = (
                f"Blocked BUY action: Drawdown={self.current_drawdown*100:.2f}% "
                f"(limit={self.config.max_drawdown_limit*100:.1f}%), "
                f"CooldownRemaining={self.cooldown_remaining} bars."
            )
            intervention = self._log_intervention(
                timestamp=timestamp,
                step=step,
                symbol=symbol,
                proposed_action="BUY",
                status=InterventionStatus.REJECTED,
                reason=reason,
                details=details,
                resulting_action="HOLD",
            )
            return 0, intervention

        # 3. Position limit check
        nav = portfolio.get_total_value({symbol: next_open})
        if nav > 0:
            current_shares = portfolio.get_position(symbol)
            current_val = current_shares * next_open
            if (current_val / nav) >= self.config.max_position_exposure:
                details = f"Position limit reached: {(current_val/nav)*100:.1f}% >= {self.config.max_position_exposure*100:.1f}%"
                intervention = self._log_intervention(
                    timestamp=timestamp,
                    step=step,
                    symbol=symbol,
                    proposed_action="BUY",
                    status=InterventionStatus.REJECTED,
                    reason=InterventionReason.POSITION_EXPOSURE_EXCEEDED,
                    details=details,
                    resulting_action="HOLD",
                )
                return 0, intervention

        return 1, None

    def _log_intervention(
        self,
        timestamp: Any,
        step: Optional[int],
        symbol: str,
        proposed_action: str,
        status: InterventionStatus,
        reason: InterventionReason,
        details: str,
        resulting_action: str,
    ) -> RiskIntervention:
        """Construct and record an auditable RiskIntervention record."""
        intervention = RiskIntervention(
            timestamp=timestamp,
            step=step,
            symbol=symbol,
            proposed_action=proposed_action,
            status=status,
            reason=reason,
            details=details,
            portfolio_nav=self.prev_nav,
            peak_nav=self.peak_nav,
            current_drawdown=self.current_drawdown,
            current_daily_loss=self.current_daily_loss,
            resulting_action=resulting_action,
        )
        if self.config.log_interventions:
            self.interventions.append(intervention)
        return intervention
