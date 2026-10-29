"""Subscription Management and Instrument Lifecycle Tracker (Phase 11B).

Manages instrument subscription states, maps application symbols to provider tokens,
handles subscription acknowledgements and errors, reconciles desired subscriptions
after reconnects, and distinguishes inactive market instruments from broken feeds.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SubscriptionState(str, Enum):
    """Lifecycle state of an individual instrument's subscription."""
    UNSUBSCRIBED = "UNSUBSCRIBED"
    DESIRED = "DESIRED"
    SUBSCRIBING = "SUBSCRIBING"
    SUBSCRIBED = "SUBSCRIBED"
    UNSUBSCRIBING = "UNSUBSCRIBING"
    ERROR = "ERROR"


class InstrumentActivityStatus(str, Enum):
    """Activity state of an instrument distinguishing illiquidity from connection failure."""
    ACTIVE = "ACTIVE"                  # Instrument is receiving regular ticks
    INACTIVE_MARKET = "INACTIVE_MARKET"  # Socket is healthy, but this specific stock has no trades
    STALE = "STALE"                    # Feed has stopped across the connection
    UNSUBSCRIBED = "UNSUBSCRIBED"      # Not currently monitored


@dataclass
class InstrumentSubscription:
    """Subscription state and telemetry for a single market instrument."""
    symbol: str
    instrument_token: Optional[int] = None
    state: SubscriptionState = SubscriptionState.UNSUBSCRIBED
    activity_status: InstrumentActivityStatus = InstrumentActivityStatus.UNSUBSCRIBED
    last_price: Optional[float] = None
    last_seen_at: Optional[str] = None
    ticks_count: int = 0
    error_message: Optional[str] = None
    subscribed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert instrument subscription to dictionary."""
        return {
            "symbol": self.symbol,
            "instrument_token": self.instrument_token,
            "state": self.state.value,
            "activity_status": self.activity_status.value,
            "last_price": self.last_price,
            "last_seen_at": self.last_seen_at,
            "ticks_count": self.ticks_count,
            "error_message": self.error_message,
            "subscribed_at": self.subscribed_at,
        }


class SubscriptionManager:
    """Thread-safe manager for instrument subscriptions, mappings, and activity tracking."""

    def __init__(
        self,
        symbol_to_token: Dict[str, int],
        max_subscriptions: int = 100,
        inactive_threshold_seconds: float = 60.0,
    ):
        self.max_subscriptions = max_subscriptions
        self.inactive_threshold_seconds = inactive_threshold_seconds
        self.symbol_to_token = dict(symbol_to_token)
        self.token_to_symbol = {v: k for k, v in self.symbol_to_token.items()}

        # Map symbol -> InstrumentSubscription
        self._instruments: Dict[str, InstrumentSubscription] = {}
        for sym, tok in self.symbol_to_token.items():
            self._instruments[sym] = InstrumentSubscription(
                symbol=sym,
                instrument_token=tok,
                state=SubscriptionState.UNSUBSCRIBED,
                activity_status=InstrumentActivityStatus.UNSUBSCRIBED,
            )

    def register_symbol(self, symbol: str, token: Optional[int] = None) -> None:
        """Register a new symbol and token mapping."""
        sym = str(symbol).strip().upper()
        if token is not None:
            self.symbol_to_token[sym] = int(token)
            self.token_to_symbol[int(token)] = sym

        if sym not in self._instruments:
            self._instruments[sym] = InstrumentSubscription(
                symbol=sym,
                instrument_token=self.symbol_to_token.get(sym),
                state=SubscriptionState.UNSUBSCRIBED,
                activity_status=InstrumentActivityStatus.UNSUBSCRIBED,
            )

    def get_token(self, symbol: str) -> Optional[int]:
        """Lookup provider instrument token for symbol."""
        return self.symbol_to_token.get(str(symbol).strip().upper())

    def get_symbol(self, token: int) -> Optional[str]:
        """Lookup canonical symbol for provider instrument token."""
        return self.token_to_symbol.get(int(token))

    def get_desired_symbols(self) -> List[str]:
        """Return symbols that are flagged as DESIRED or SUBSCRIBED."""
        res: List[str] = []
        for sym, sub in self._instruments.items():
            if sub.state in (SubscriptionState.DESIRED, SubscriptionState.SUBSCRIBING, SubscriptionState.SUBSCRIBED):
                res.append(sym)
        return sorted(res)

    def get_subscribed_symbols(self) -> List[str]:
        """Return symbols currently in SUBSCRIBED or SUBSCRIBING state."""
        return sorted([
            sym for sym, sub in self._instruments.items()
            if sub.state in (SubscriptionState.SUBSCRIBED, SubscriptionState.SUBSCRIBING)
        ])

    def mark_desired(self, symbols: List[str]) -> Tuple[List[str], Optional[str]]:
        """Mark symbols as desired to subscribe, checking limits.
        
        Returns:
            Tuple of (symbols_to_subscribe_on_wire, error_message).
        """
        to_add: List[str] = []
        clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]

        current_active = len(self.get_subscribed_symbols())
        for sym in clean_symbols:
            self.register_symbol(sym)
            sub = self._instruments[sym]
            if sub.state not in (SubscriptionState.SUBSCRIBED, SubscriptionState.SUBSCRIBING):
                if current_active + len(to_add) >= self.max_subscriptions:
                    err = f"Subscription limit exceeded: maximum {self.max_subscriptions} symbols allowed."
                    return to_add, err
                sub.state = SubscriptionState.DESIRED
                sub.error_message = None
                to_add.append(sym)

        return to_add, None

    def mark_subscribing(self, symbols: List[str]) -> None:
        """Mark symbols as in-flight wire request sent."""
        for s in symbols:
            sym = str(s).strip().upper()
            if sym in self._instruments:
                self._instruments[sym].state = SubscriptionState.SUBSCRIBING

    def mark_subscribed(self, symbols: List[str]) -> None:
        """Confirm subscription active (via acknowledgement or wire success)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        for s in symbols:
            sym = str(s).strip().upper()
            if sym in self._instruments:
                sub = self._instruments[sym]
                sub.state = SubscriptionState.SUBSCRIBED
                sub.subscribed_at = now_iso
                sub.error_message = None
                if sub.activity_status == InstrumentActivityStatus.UNSUBSCRIBED:
                    sub.activity_status = InstrumentActivityStatus.ACTIVE

    def mark_unsubscribed(self, symbols: List[str]) -> None:
        """Mark symbols as unsubscribed."""
        for s in symbols:
            sym = str(s).strip().upper()
            if sym in self._instruments:
                sub = self._instruments[sym]
                sub.state = SubscriptionState.UNSUBSCRIBED
                sub.activity_status = InstrumentActivityStatus.UNSUBSCRIBED
                sub.error_message = None

    def mark_error(self, symbols: List[str], error_message: str) -> None:
        """Mark symbols as subscription failed."""
        for s in symbols:
            sym = str(s).strip().upper()
            if sym in self._instruments:
                sub = self._instruments[sym]
                sub.state = SubscriptionState.ERROR
                sub.error_message = str(error_message)

    def handle_acknowledgement(self, ack: Dict[str, Any]) -> None:
        """Handle KiteTicker JSON control message acknowledgement."""
        msg_type = ack.get("type")
        status = ack.get("status", True)
        tokens = ack.get("data") or []

        if not isinstance(tokens, list):
            tokens = [tokens]

        symbols = [self.get_symbol(t) for t in tokens if self.get_symbol(t) is not None]

        if msg_type == "subscribe":
            if status:
                self.mark_subscribed(symbols)
            else:
                self.mark_error(symbols, ack.get("message") or "Subscription rejected by broker.")
        elif msg_type == "unsubscribe":
            if status:
                self.mark_unsubscribed(symbols)
            else:
                self.mark_error(symbols, ack.get("message") or "Unsubscription rejected by broker.")
        elif msg_type == "error":
            err_msg = ack.get("message") or "Broker WebSocket error."
            # Apply error to all in-flight or all subscribed instruments
            for sub in self._instruments.values():
                if sub.state == SubscriptionState.SUBSCRIBING:
                    sub.state = SubscriptionState.ERROR
                    sub.error_message = err_msg

    def record_tick(self, symbol: str, price: float, timestamp: str) -> None:
        """Record verified incoming tick for instrument."""
        sym = str(symbol).strip().upper()
        self.register_symbol(sym)
        sub = self._instruments[sym]
        sub.last_price = price
        sub.last_seen_at = timestamp
        sub.ticks_count += 1
        sub.activity_status = InstrumentActivityStatus.ACTIVE

    def reconcile_after_reconnect(self) -> List[str]:
        """Reconcile and return list of symbols that must be resubscribed after a reconnection.
        
        Preserves desired portfolio subscriptions and marks them as SUBSCRIBING.
        """
        resubscribe_symbols: List[str] = []
        for sym, sub in self._instruments.items():
            if sub.state in (SubscriptionState.SUBSCRIBED, SubscriptionState.SUBSCRIBING, SubscriptionState.DESIRED):
                sub.state = SubscriptionState.SUBSCRIBING
                resubscribe_symbols.append(sym)
        return sorted(resubscribe_symbols)

    def evaluate_instrument_activity(self, is_connection_alive: bool) -> None:
        """Update activity status of instruments, distinguishing inactive market from dead socket.
        
        Logic:
            - If connection is dead: subscribed instruments without updates become STALE.
            - If connection is alive: subscribed instruments without updates become INACTIVE_MARKET
              (i.e. socket is receiving messages, but this stock has no current volume / off-hours).
        """
        now = datetime.now(timezone.utc)
        for sym, sub in self._instruments.items():
            if sub.state != SubscriptionState.SUBSCRIBED:
                continue

            if not sub.last_seen_at:
                # Subscribed, but no ticks ever seen
                sub.activity_status = (
                    InstrumentActivityStatus.INACTIVE_MARKET if is_connection_alive
                    else InstrumentActivityStatus.STALE
                )
                continue

            try:
                last_dt = datetime.fromisoformat(sub.last_seen_at.replace("Z", "+00:00"))
                elapsed = (now - last_dt).total_seconds()
                if elapsed > self.inactive_threshold_seconds:
                    sub.activity_status = (
                        InstrumentActivityStatus.INACTIVE_MARKET if is_connection_alive
                        else InstrumentActivityStatus.STALE
                    )
                else:
                    sub.activity_status = InstrumentActivityStatus.ACTIVE
            except Exception:
                pass

    def get_subscription_details(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic summary of all instrument subscriptions."""
        details = {sym: sub.to_dict() for sym, sub in sorted(self._instruments.items())}
        active_count = sum(1 for s in self._instruments.values() if s.activity_status == InstrumentActivityStatus.ACTIVE)
        inactive_count = sum(1 for s in self._instruments.values() if s.activity_status == InstrumentActivityStatus.INACTIVE_MARKET)
        subscribed_count = len(self.get_subscribed_symbols())

        return {
            "total_registered": len(self._instruments),
            "subscribed_count": subscribed_count,
            "active_instruments": active_count,
            "inactive_market_instruments": inactive_count,
            "max_subscriptions": self.max_subscriptions,
            "instruments": details,
        }
