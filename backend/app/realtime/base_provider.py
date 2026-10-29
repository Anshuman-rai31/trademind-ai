"""Abstract Base Provider for Real-Time Market Data Feeds (Phase 11A / 11B).

Defines the pluggable interface contract for market-data streaming providers
(e.g., Zerodha Kite Connect, mock testing feeds, broker WebSockets), including
WebSocket lifecycle management, subscription state reconciliation, and message handling.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .models import ConnectionStatus, ConnectionTelemetry, MarketQuote, ProviderConfig
from .subscription import InstrumentActivityStatus, SubscriptionManager, SubscriptionState
from .validator import RealtimeDataValidator

logger = logging.getLogger(__name__)


class RealtimeDataProvider(ABC):
    """Abstract base class for all real-time market data providers."""

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        validator: Optional[RealtimeDataValidator] = None,
        subscription_manager: Optional[SubscriptionManager] = None,
    ):
        self.config = config or ProviderConfig()
        self.validator = validator or RealtimeDataValidator()
        self.subscription_manager = subscription_manager or SubscriptionManager(
            symbol_to_token={},
            max_subscriptions=self.config.max_subscriptions,
        )
        self._telemetry = ConnectionTelemetry(
            status=ConnectionStatus.DISCONNECTED,
            provider_name=self.config.provider_name,
        )
        self._quotes_cache: Dict[str, MarketQuote] = {}
        self._callbacks: List[Callable[[MarketQuote], None]] = []

    @property
    def status(self) -> ConnectionStatus:
        """Current provider connection lifecycle state."""
        return self._telemetry.status

    @property
    def telemetry(self) -> ConnectionTelemetry:
        """Read-only copy of connection diagnostic telemetry."""
        return self._telemetry

    @property
    def subscribed_symbols(self) -> List[str]:
        """List of currently subscribed symbols."""
        return self.subscription_manager.get_subscribed_symbols()

    def register_callback(self, callback: Callable[[MarketQuote], None]) -> None:
        """Register a subscriber callback invoked on each validated market quote."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[MarketQuote], None]) -> None:
        """Unregister a subscriber callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_latest_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Retrieve latest cached validated quote for a specific instrument."""
        return self._quotes_cache.get(str(symbol).strip().upper())

    def get_all_quotes(self) -> Dict[str, MarketQuote]:
        """Retrieve copy of all latest cached quotes."""
        return dict(self._quotes_cache)

    def get_subscription_details(self) -> Dict[str, Any]:
        """Retrieve detailed subscription registry and instrument activity."""
        is_alive = self._telemetry.status == ConnectionStatus.CONNECTED and not self._telemetry.is_stale
        self.subscription_manager.evaluate_instrument_activity(is_connection_alive=is_alive)
        return self.subscription_manager.get_subscription_details()

    def _dispatch_quote(self, quote: MarketQuote) -> None:
        """Update internal cache, subscription tracker, and notify registered subscriber callbacks."""
        self._quotes_cache[quote.symbol] = quote
        self.subscription_manager.record_tick(
            symbol=quote.symbol,
            price=quote.last_price,
            timestamp=quote.timestamp,
        )
        for cb in self._callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.error("Error in realtime quote callback: %s", exc)

    @abstractmethod
    def connect(self) -> bool:
        """Initiate connection and start background streaming.
        
        Returns:
            bool indicating if connection was successfully initiated.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect and terminate background listener threads."""
        pass

    @abstractmethod
    def authenticate(self) -> bool:
        """Verify credentials and session permissions with the provider.
        
        Returns:
            bool indicating if authentication was successful.
        """
        pass

    @abstractmethod
    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to streaming market updates for a list of instruments.
        
        Returns:
            bool indicating if subscription request was successfully processed.
        """
        pass

    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from streaming updates for a list of instruments.
        
        Returns:
            bool indicating if unsubscription request was successfully processed.
        """
        pass
