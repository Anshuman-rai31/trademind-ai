"""TradeMind Phase 5 Reinforcement Learning Actions.

Canonical discrete action definitions for Gymnasium trading environments.
"""
from enum import IntEnum


class TradingAction(IntEnum):
    """Discrete trading action space for reinforcement learning agent.
    
    Attributes:
        HOLD: 0 - Maintain current position; execute zero trades.
        BUY:  1 - Enter long position using allocated cash buffer.
        SELL: 2 - Exit long position and return to 100% unencumbered cash.
    """
    HOLD = 0
    BUY = 1
    SELL = 2
