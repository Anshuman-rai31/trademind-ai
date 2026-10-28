"""Risk Management Service for TradeMind (Phase 9).

Integrates the Phase 7 RiskManager engine into API endpoints for pre-trade risk
inspection, position constraint validation, and policy audit reporting.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..simulation.models import Order, OrderSide
from ..simulation.portfolio import Portfolio


class RiskService:
    """Service layer interfacing with RiskManager."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.risk_manager = RiskManager(config=self.config)

    def get_config(self) -> Dict[str, Any]:
        """Return active risk management configuration dictionary."""
        return self.config.to_dict()

    def get_status(self) -> Dict[str, Any]:
        """Return operational state and capability status of the risk engine."""
        return {
            "status": "operational",
            "enabled": self.config.enabled,
            "live_portfolio_active": False,
            "note": "TradeMind operates in research and simulation mode. No live exchange order routing.",
            "configuration": self.config.to_dict(),
            "interventions_recorded_in_session": len(self.risk_manager.interventions),
            "is_restricted": self.risk_manager.is_restricted,
            "cooldown_remaining_bars": self.risk_manager.cooldown_remaining,
        }

    def evaluate_proposed_action(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a proposed trade against risk constraints."""
        clean_sym = symbol.strip().upper()
        clean_side_str = side.strip().upper()

        if clean_side_str not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")

        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        side_enum = OrderSide.BUY if clean_side_str == "BUY" else OrderSide.SELL

        # Construct portfolio state
        p_state = portfolio_state or {}
        cash = float(p_state.get("cash", 100000.0))
        portfolio = Portfolio(initial_cash=cash)

        holdings = p_state.get("holdings", {})
        if isinstance(holdings, dict):
            for sym, qty in holdings.items():
                portfolio.holdings[sym.upper()] = float(qty)

        now_ts = pd.Timestamp.now(tz="UTC")
        order = Order(
            symbol=clean_sym,
            side=side_enum,
            quantity=float(quantity),
            created_at=now_ts,
        )

        approved_order, intervention = self.risk_manager.evaluate_proposed_order(
            order=order,
            portfolio=portfolio,
            execution_price=float(price),
            timestamp=now_ts,
        )

        requested_action = {
            "symbol": clean_sym,
            "side": clean_side_str,
            "quantity": float(quantity),
            "price": float(price),
            "estimated_value": float(quantity * price),
        }

        if approved_order is not None:
            effective_action = {
                "symbol": approved_order.symbol,
                "side": approved_order.side.value,
                "quantity": float(approved_order.quantity),
                "price": float(price),
                "estimated_value": float(approved_order.quantity * price),
            }
        else:
            effective_action = None

        return {
            "requested_action": requested_action,
            "effective_action": effective_action,
            "approved": approved_order is not None,
            "intervention_occurred": intervention is not None,
            "intervention_reason": intervention.reason.value if intervention else None,
            "intervention_status": intervention.status.value if intervention else None,
            "details": intervention.details if intervention else "Order conforms with risk parameters.",
        }
