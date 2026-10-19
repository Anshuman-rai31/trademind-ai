"""TradeMind Phase 4 Baseline Strategies Package."""

from .base import BaseStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .ma_crossover import MACrossoverStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
]
