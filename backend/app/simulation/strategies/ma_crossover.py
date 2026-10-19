"""TradeMind Phase 4 Moving Average Crossover Strategy.

Generates systematic trend-following signals based on fast and slow moving
average crossovers. Enforces strict zero-future-leakage by evaluating signals
only on historical close prices at bar t, dispatching orders to execute at bar t+1.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from ..models import Order, OrderSide
from ..portfolio import Portfolio
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover trading strategy.
    
    Generates a BUY signal when the fast moving average crosses strictly above
    the slow moving average (Golden Cross), and a SELL signal when the fast MA
    crosses strictly below the slow MA (Death Cross).
    """

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 50,
        ma_type: str = "SMA",
        target_symbol: Optional[str] = None,
        allocation_pct: float = 0.95,
        name: str = "MACrossover",
    ) -> None:
        """Initialize Moving Average Crossover Strategy.
        
        Args:
            fast_window: Lookback period for fast moving average (e.g. 10 bars).
            slow_window: Lookback period for slow moving average (e.g. 50 bars).
            ma_type: 'SMA' for Simple Moving Average or 'EMA' for Exponential.
            target_symbol: Symbol to trade (defaults to first configured symbol).
            allocation_pct: Fraction of available cash to allocate on buy signals (default 0.95).
            name: Strategy display identifier.
        """
        super().__init__(name=name)
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError(f"Windows must be positive, got fast={fast_window}, slow={slow_window}")
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})"
            )
        if ma_type.upper() not in ("SMA", "EMA"):
            raise ValueError(f"ma_type must be 'SMA' or 'EMA', got {ma_type}")
        if not (0.0 < allocation_pct <= 1.0):
            raise ValueError(f"allocation_pct must be in (0, 1], got {allocation_pct}")

        self.fast_window: int = fast_window
        self.slow_window: int = slow_window
        self.ma_type: str = ma_type.upper()
        self.target_symbol: Optional[str] = target_symbol
        self.allocation_pct: float = float(allocation_pct)

    def reset(self) -> None:
        """Reset state before a new simulation run."""
        pass

    def generate_orders(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Order]:
        """Generate crossover orders based on data available up to bar t.
        
        Args:
            bar_idx: Current bar index t.
            timestamp: Current bar timestamp.
            historical_data: Historical OHLCV data slice [0 : bar_idx + 1].
            portfolio: Current portfolio state.
            
        Returns:
            List containing generated Order, or empty list if no crossover occurred.
        """
        symbol = self.target_symbol or (self.symbols[0] if self.symbols else None)
        if not symbol or symbol not in historical_data:
            return []

        df = historical_data[symbol]
        # Must have at least slow_window + 1 bars to detect a crossover transition
        if len(df) < self.slow_window + 1:
            return []

        close_series = df["close"]

        # Compute moving averages on historical slice
        if self.ma_type == "SMA":
            fast_ma = close_series.rolling(self.fast_window).mean()
            slow_ma = close_series.rolling(self.slow_window).mean()
        else:  # EMA
            fast_ma = close_series.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = close_series.ewm(span=self.slow_window, adjust=False).mean()

        # Last two values: t (current) and t-1 (previous)
        fast_curr = float(fast_ma.iloc[-1])
        fast_prev = float(fast_ma.iloc[-2])
        slow_curr = float(slow_ma.iloc[-1])
        slow_prev = float(slow_ma.iloc[-2])

        if pd.isna(fast_curr) or pd.isna(fast_prev) or pd.isna(slow_curr) or pd.isna(slow_prev):
            return []

        # Crossover Detection
        golden_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        death_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)

        current_position = portfolio.get_position(symbol)

        # Bullish signal & currently FLAT: enter long position
        if golden_cross and current_position <= 0:
            assert self.config is not None, "Strategy not initialized with SimulationConfig"
            curr_close = float(close_series.iloc[-1])
            if curr_close <= 0:
                return []

            est_price = curr_close * (1.0 + self.config.slippage_rate)
            fee_factor = 1.0 + self.config.fee_rate
            cash_to_invest = portfolio.cash * self.allocation_pct
            shares = cash_to_invest / (est_price * fee_factor)

            if not self.config.allow_fractional:
                shares = float(int(shares))

            if shares >= self.config.min_trade_size:
                logger.debug(
                    "[%s] Golden Cross BUY signal for %s: %s shares at approx %s",
                    timestamp,
                    symbol,
                    shares,
                    curr_close,
                )
                return [
                    Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=shares,
                        created_at=timestamp,
                    )
                ]

        # Bearish signal & currently LONG: liquidate position
        elif death_cross and current_position > 0:
            logger.debug(
                "[%s] Death Cross SELL signal for %s: liquidating %s shares",
                timestamp,
                symbol,
                current_position,
            )
            return [
                Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_position,
                    created_at=timestamp,
                )
            ]

        return []
