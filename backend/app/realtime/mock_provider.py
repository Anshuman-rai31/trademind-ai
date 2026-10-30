"""Mock Real-Time Market Data Provider for Testing and Simulation (Phase 11A / 11B).

Simulates streaming NSE quotes with controllable latency, heartbeat telemetry,
error injection (auth failure, staleness, network disconnects), binary frame support,
subscription state tracking, and deterministic feeds.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base_provider import RealtimeDataProvider
from .models import ConnectionStatus, ConnectionTelemetry, MarketQuote, ProviderConfig
from .parser import KiteProtocolParser
from .subscription import InstrumentActivityStatus, SubscriptionManager, SubscriptionState
from .validator import RealtimeDataValidator

logger = logging.getLogger(__name__)

# Baseline realistic prices for supported NSE equities
MOCK_NSE_PRICES: Dict[str, float] = {
    "RELIANCE.NS": 2850.50,
    "TCS.NS": 3950.00,
    "INFY.NS": 1620.25,
    "ICICIBANK.NS": 1180.75,
    "SBIN.NS": 790.10,
}

MOCK_TOKEN_MAP: Dict[str, int] = {
    "RELIANCE.NS": 738561,
    "TCS.NS": 2953217,
    "INFY.NS": 408065,
    "ICICIBANK.NS": 1270529,
    "SBIN.NS": 779521,
}


class MockRealtimeProvider(RealtimeDataProvider):
    """Mock real-time provider for offline simulation and test suites."""

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        validator: Optional[RealtimeDataValidator] = None,
        simulate_auth_failure: bool = False,
        simulate_connection_error: bool = False,
        auto_tick_generation: bool = False,
        token_map: Optional[Dict[str, int]] = None,
    ):
        cfg = config or ProviderConfig(provider_name="mock", api_key="mock_key", access_token="mock_token")
        token_mapping = dict(token_map or MOCK_TOKEN_MAP)
        sub_mgr = SubscriptionManager(
            symbol_to_token=token_mapping,
            max_subscriptions=cfg.max_subscriptions,
            inactive_threshold_seconds=cfg.stale_timeout_seconds * 2.0,
        )
        super().__init__(config=cfg, validator=validator, subscription_manager=sub_mgr)

        self.simulate_auth_failure = simulate_auth_failure
        self.simulate_connection_error = simulate_connection_error
        self.auto_tick_generation = auto_tick_generation
        self._current_prices = dict(MOCK_NSE_PRICES)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._connect_lock = threading.Lock()

    def authenticate(self) -> bool:
        """Simulate authentication validation."""
        if self.simulate_auth_failure:
            self._telemetry.status = ConnectionStatus.AUTH_FAILED
            self._telemetry.last_error = "Mock authentication failure simulated."
            return False
        return True

    def connect(self) -> bool:
        """Connect to mock streaming feed, preventing concurrent duplicate connections."""
        with self._connect_lock:
            if self._telemetry.status == ConnectionStatus.CONNECTED:
                return True

            if self._telemetry.status == ConnectionStatus.CONNECTING:
                return True

            self._telemetry.status = ConnectionStatus.CONNECTING
            self._telemetry.last_error = None

            if not self.authenticate():
                return False

            if self.simulate_connection_error:
                self._telemetry.status = ConnectionStatus.ERROR
                self._telemetry.last_error = "Mock connection failure simulated."
                return False

            self._is_running = True
            self._telemetry.status = ConnectionStatus.CONNECTED
            now_iso = datetime.now(timezone.utc).isoformat()
            self._telemetry.connected_at = now_iso
            self._telemetry.last_heartbeat_at = now_iso
            self._telemetry.reconnect_attempts = 0

            # Reconcile subscriptions on reconnect
            self.reconcile_subscriptions()

            if self.auto_tick_generation:
                if self._thread is None or not self._thread.is_alive():
                    self._start_tick_loop()
                if self.subscribed_symbols:
                    self.emit_mock_ticks_for_subscribed()

            return True

    def disconnect(self) -> None:
        """Cleanly terminate mock stream and background worker threads."""
        with self._connect_lock:
            self._is_running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                self._thread = None
            self._telemetry.status = ConnectionStatus.DISCONNECTED
            self._telemetry.is_stale = False

    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe mock feed to symbols with limit enforcement."""
        to_add, err = self.subscription_manager.mark_desired(symbols)
        if err:
            self._telemetry.last_error = err
            return False

        if not to_add:
            return True

        self.subscription_manager.mark_subscribing(to_add)
        self.subscription_manager.mark_subscribed(to_add)
        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        if self.auto_tick_generation and self.status == ConnectionStatus.CONNECTED:
            self.emit_mock_ticks_for_subscribed()
        return True

    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        self.subscription_manager.mark_unsubscribed(clean_symbols)
        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        return True

    def reconcile_subscriptions(self) -> bool:
        """Reconcile and mark all desired subscriptions active upon reconnect."""
        resub_symbols = self.subscription_manager.reconcile_after_reconnect()
        if resub_symbols:
            self.subscription_manager.mark_subscribed(resub_symbols)
        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        return True

    def push_binary_frame(self, data: bytes) -> Tuple[int, int]:
        """Push a raw binary frame through parser and validation."""
        self._telemetry.messages_received += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_message_at = now_iso
        self._telemetry.last_heartbeat_at = now_iso
        self._telemetry.is_stale = False

        ticks, err = KiteProtocolParser.parse_binary_message(data)
        if err:
            self._telemetry.messages_rejected += 1

        valid_count = 0
        rejected_count = 0

        for raw_tick in ticks:
            tok = raw_tick.get("instrument_token")
            sym = self.subscription_manager.get_symbol(int(tok)) if tok else None
            ok, quote, _ = self.validator.validate_message(raw_msg=raw_tick, symbol_override=sym, source="mock")
            if ok and quote:
                self._dispatch_quote(quote)
                valid_count += 1
            else:
                self._telemetry.messages_rejected += 1
                rejected_count += 1

        return valid_count, rejected_count

    def push_text_frame(self, text: str) -> Dict[str, Any]:
        """Push a text/JSON frame through control parser."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_heartbeat_at = now_iso

        parsed = KiteProtocolParser.parse_text_message(text)
        msg_type = parsed.get("type")
        if msg_type in ("subscribe", "unsubscribe", "error"):
            self.subscription_manager.handle_acknowledgement(parsed)
            if msg_type == "error":
                self._telemetry.last_error = parsed.get("message")
        return parsed

    def push_tick(self, raw_tick: Dict[str, Any]) -> Tuple[bool, Optional[MarketQuote]]:
        """Directly push a raw tick dictionary through validation and dispatch."""
        self._telemetry.messages_received += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_message_at = now_iso
        self._telemetry.last_heartbeat_at = now_iso
        self._telemetry.is_stale = False

        is_valid, quote, err = self.validator.validate_message(
            raw_msg=raw_tick,
            source="mock",
        )

        if not is_valid or quote is None:
            self._telemetry.messages_rejected += 1
            return False, None

        self._dispatch_quote(quote)
        return True, quote

    def emit_mock_ticks_for_subscribed(self) -> List[MarketQuote]:
        """Emit one valid tick for each currently subscribed symbol."""
        now_dt = datetime.now(timezone.utc)
        quotes: List[MarketQuote] = []
        for sym in list(self.subscribed_symbols):
            base_p = self._current_prices.get(sym, 1000.0)
            import random
            drift = random.uniform(-0.002, 0.002)
            new_p = round(base_p * (1.0 + drift), 2)
            self._current_prices[sym] = new_p

            tick = {
                "symbol": sym,
                "last_price": new_p,
                "timestamp": now_dt.isoformat(),
                "volume": int(random.randint(100, 5000)),
                "bid": round(new_p * 0.9998, 2),
                "ask": round(new_p * 1.0002, 2),
            }
            ok, quote = self.push_tick(tick)
            if ok and quote:
                quotes.append(quote)
        return quotes

    def _start_tick_loop(self) -> None:
        """Start background daemon thread generating mock ticks."""
        def run():
            while self._is_running:
                if self._telemetry.status == ConnectionStatus.CONNECTED and self.subscribed_symbols:
                    self.emit_mock_ticks_for_subscribed()
                time.sleep(1.0)

        self._thread = threading.Thread(target=run, daemon=True, name="MockRealtimeFeedThread")
        self._thread.start()
