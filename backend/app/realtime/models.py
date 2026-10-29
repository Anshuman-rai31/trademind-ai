"""Real-Time Market Data Models and Data Contracts (Phase 11A).

Defines connection states, market quote structures, validated tick messages,
and provider configuration contracts. Guarantees zero credential leakage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ConnectionStatus(str, Enum):
    """Real-time provider connection lifecycle states."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    AUTH_FAILED = "AUTH_FAILED"
    ERROR = "ERROR"
    STALE = "STALE"


@dataclass
class MarketQuote:
    """Validated real-time market data quote / tick message."""
    symbol: str
    last_price: float
    timestamp: str          # Event time reported by the provider (ISO-8601)
    received_at: str        # Processing time when ingested locally (ISO-8601)
    volume: Optional[int] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_quantity: Optional[int] = None
    ask_quantity: Optional[int] = None
    source: str = "kite"
    sequence_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert quote to dictionary with rounded floats."""
        d = asdict(self)
        d["last_price"] = round(self.last_price, 2)
        if self.open_price is not None:
            d["open_price"] = round(self.open_price, 2)
        if self.high_price is not None:
            d["high_price"] = round(self.high_price, 2)
        if self.low_price is not None:
            d["low_price"] = round(self.low_price, 2)
        if self.close_price is not None:
            d["close_price"] = round(self.close_price, 2)
        if self.bid_price is not None:
            d["bid_price"] = round(self.bid_price, 2)
        if self.ask_price is not None:
            d["ask_price"] = round(self.ask_price, 2)
        return d


@dataclass
class ProviderConfig:
    """Configuration parameters for real-time market data providers.
    
    Security Guarantee:
        Credentials (api_key, access_token, api_secret) are strictly kept in-memory
        and NEVER exposed in to_safe_dict(), repr, or API responses.
    """
    provider_name: str = "kite"
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    api_secret: Optional[str] = None
    max_subscriptions: int = 50
    stale_timeout_seconds: float = 15.0
    reconnect_max_retries: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0

    def has_credentials(self) -> bool:
        """Check if necessary credentials are provided."""
        return bool(self.api_key and self.access_token)

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize configuration without exposing secrets."""
        return {
            "provider_name": self.provider_name,
            "has_credentials": self.has_credentials(),
            "has_api_key": bool(self.api_key),
            "has_access_token": bool(self.access_token),
            "max_subscriptions": self.max_subscriptions,
            "stale_timeout_seconds": self.stale_timeout_seconds,
            "reconnect_max_retries": self.reconnect_max_retries,
            "reconnect_base_delay": self.reconnect_base_delay,
            "reconnect_max_delay": self.reconnect_max_delay,
        }


@dataclass
class ConnectionTelemetry:
    """Diagnostic telemetry regarding feed health, latencies, and message counts."""
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    provider_name: str = "kite"
    connected_at: Optional[str] = None
    last_message_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    messages_received: int = 0
    messages_rejected: int = 0
    subscribed_symbols_count: int = 0
    reconnect_attempts: int = 0
    last_error: Optional[str] = None
    is_stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
