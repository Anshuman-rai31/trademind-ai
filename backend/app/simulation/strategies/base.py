"""TradeMind Phase 4 Abstract Strategy Interface.

Base class for all research and baseline trading strategies.
Strictly separates signal generation at time t from execution at time t+1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from ..models import Order, SimulationConfig
from ..portfolio import Portfolio


class BaseStrategy(ABC):
    """Abstract trading strategy class.
    
    Subclasses implement generate_orders() which is invoked at the close
    of bar t. The strategy inspects only historical and current bar information
    (<= t) and returns a list of orders to be filled at bar t+1 open.
    """

    def __init__(self, name: str = "BaseStrategy") -> None:
        self.name: str = name
        self.config: Optional[SimulationConfig] = None
        self.symbols: List[str] = []

    def initialize(self, config: SimulationConfig, symbols: List[str]) -> None:
        """Called by the simulator prior to running the backtest.
        
        Args:
            config: Active simulation configuration parameters.
            symbols: List of traded tickers in this backtest.
        """
        self.config = config
        self.symbols = symbols
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset internal strategy state for a fresh backtest run."""
        pass

    @abstractmethod
    def generate_orders(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Order]:
        """Generate orders based strictly on information available at time t.
        
        CRITICAL NO-LOOKAHEAD CONTRACT:
            historical_data contains DataFrame slices up to and including
            index bar_idx. Any attempt to access data beyond bar_idx or future
            prices must be rejected.
            
        Args:
            bar_idx: Zero-based integer index of the current valuation bar t.
            timestamp: Timestamp of the current bar t.
            historical_data: Map of symbol -> DataFrame containing bars [0 : bar_idx + 1].
            portfolio: Current portfolio state as of bar t close.
            
        Returns:
            List of Order instances to be queued for execution at bar t+1 open.
        """
        pass
