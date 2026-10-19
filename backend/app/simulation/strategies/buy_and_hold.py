"""TradeMind Phase 4 Buy-and-Hold Baseline Strategy.

Buys the target asset at the onset of the evaluation period and holds it
continuously through the backtest horizon, accounting for realistic transaction
fees, slippage, and daily mark-to-market valuations.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from ..models import Order, OrderSide
from ..portfolio import Portfolio
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(BaseStrategy):
    """Buy-and-hold benchmark strategy.
    
    Generates a single BUY order at the first available trading bar for the
    configured allocation proportion, then holds the position continuously.
    """

    def __init__(
        self,
        target_symbol: Optional[str] = None,
        allocation_pct: float = 0.98,
        name: str = "BuyAndHold",
    ) -> None:
        """Initialize BuyAndHold strategy.
        
        Args:
            target_symbol: Symbol to purchase (defaults to first configured symbol).
            allocation_pct: Fraction of available cash to allocate (default 0.98,
                            leaving a 2% buffer for slippage and brokerage fees).
            name: Strategy display identifier.
        """
        super().__init__(name=name)
        if not (0.0 < allocation_pct <= 1.0):
            raise ValueError(f"allocation_pct must be in (0, 1], got {allocation_pct}")
        self.target_symbol: Optional[str] = target_symbol
        self.allocation_pct: float = float(allocation_pct)
        self._has_bought: bool = False

    def reset(self) -> None:
        """Reset state before a new simulation run."""
        self._has_bought = False

    def generate_orders(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Order]:
        """Generate a single BUY order on the first available bar.
        
        Args:
            bar_idx: Current bar index t.
            timestamp: Current bar timestamp.
            historical_data: Historical OHLCV data up to bar t.
            portfolio: Current portfolio state.
            
        Returns:
            List containing one BUY order on bar 0, or empty list afterwards.
        """
        if self._has_bought:
            return []

        symbol = self.target_symbol or (self.symbols[0] if self.symbols else None)
        if not symbol or symbol not in historical_data:
            return []

        df = historical_data[symbol]
        if df.empty or bar_idx >= len(df):
            return []

        # Current price at close of bar t used as proxy for estimating affordable shares
        close_price = float(df.iloc[bar_idx]["close"])
        if close_price <= 0 or pd.isna(close_price):
            return []

        assert self.config is not None, "Strategy not initialized with SimulationConfig"

        # Buffer calculation: price + slippage + fee
        est_price = close_price * (1.0 + self.config.slippage_rate)
        fee_factor = 1.0 + self.config.fee_rate
        cash_to_invest = portfolio.cash * self.allocation_pct
        shares = cash_to_invest / (est_price * fee_factor)

        if not self.config.allow_fractional:
            shares = float(int(shares))

        if shares < self.config.min_trade_size:
            logger.warning(
                "Affordable shares (%s) below min_trade_size (%s). Cash: %s",
                shares,
                self.config.min_trade_size,
                portfolio.cash,
            )
            return []

        self._has_bought = True
        return [
            Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=shares,
                created_at=timestamp,
            )
        ]
