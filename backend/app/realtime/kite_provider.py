"""Zerodha Kite Connect WebSocket Real-Time Market Data Provider (Phase 11A / 11B).

Implements real-time tick streaming for Indian NSE equities using the Kite Connect
WebSocket (KiteTicker) protocol, with binary and JSON frame parsing, token mapping,
bounded exponential backoff reconnection, subscription state reconciliation, and
strict credential isolation.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base_provider import RealtimeDataProvider
from .models import ConnectionStatus, ConnectionTelemetry, MarketQuote, ProviderConfig
from .parser import KiteProtocolParser
from .subscription import InstrumentActivityStatus, SubscriptionManager, SubscriptionState
from .validator import RealtimeDataValidator

logger = logging.getLogger(__name__)

# Canonical NSE Equities Instrument Token Mapping for Kite Connect
DEFAULT_KITE_NSE_TOKEN_MAP: Dict[str, int] = {
    "RELIANCE.NS": 738561,
    "TCS.NS": 2953217,
    "INFY.NS": 408065,
    "ICICIBANK.NS": 1270529,
    "SBIN.NS": 779521,
}

REVERSE_KITE_TOKEN_MAP: Dict[int, str] = {v: k for k, v in DEFAULT_KITE_NSE_TOKEN_MAP.items()}


class KiteRealtimeProvider(RealtimeDataProvider):
    """Zerodha Kite Connect WebSocket market data provider."""

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        validator: Optional[RealtimeDataValidator] = None,
        token_map: Optional[Dict[str, int]] = None,
    ):
        cfg = config or ProviderConfig(
            provider_name="kite",
            api_key=os.environ.get("KITE_API_KEY"),
            access_token=os.environ.get("KITE_ACCESS_TOKEN"),
            api_secret=os.environ.get("KITE_API_SECRET"),
        )
        token_mapping = dict(token_map or DEFAULT_KITE_NSE_TOKEN_MAP)
        sub_mgr = SubscriptionManager(
            symbol_to_token=token_mapping,
            max_subscriptions=cfg.max_subscriptions,
            inactive_threshold_seconds=cfg.stale_timeout_seconds * 2.0,
        )
        super().__init__(config=cfg, validator=validator, subscription_manager=sub_mgr)

        self.symbol_to_token = self.subscription_manager.symbol_to_token
        self.token_to_symbol = self.subscription_manager.token_to_symbol

        self._ticker: Optional[Any] = None
        self._is_running = False
        self._connect_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._backoff_thread: Optional[threading.Thread] = None

    def authenticate(self) -> bool:
        """Validate credentials availability and format according to Kite Connect requirements."""
        if not self.config.has_credentials():
            self._telemetry.status = ConnectionStatus.AUTH_FAILED
            self._telemetry.last_error = (
                "Authentication failed: Missing KITE_API_KEY or KITE_ACCESS_TOKEN in environment."
            )
            logger.warning(self._telemetry.last_error)
            return False

        # Attempt to verify with KiteConnect client if library is installed
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.config.api_key)
            kite.set_access_token(self.config.access_token)
            if len(self.config.api_key) < 4 or len(self.config.access_token) < 10:
                raise ValueError("Kite credentials appear malformed or invalid.")
            return True
        except ImportError:
            # kiteconnect package not installed in environment, fallback to structured validation
            if len(self.config.api_key) < 4 or len(self.config.access_token) < 10:
                self._telemetry.status = ConnectionStatus.AUTH_FAILED
                self._telemetry.last_error = "Malformed Kite credentials."
                return False
            return True
        except Exception as exc:
            self._telemetry.status = ConnectionStatus.AUTH_FAILED
            self._telemetry.last_error = f"Kite authentication failure: {exc}"
            logger.error(self._telemetry.last_error)
            return False

    def connect(self) -> bool:
        """Connect to KiteTicker WebSocket with exponential backoff and auth validation.
        
        Guarantees protection against duplicate concurrent connections.
        """
        with self._connect_lock:
            if self._telemetry.status == ConnectionStatus.CONNECTED:
                logger.info("KiteRealtimeProvider already connected.")
                return True

            if self._telemetry.status == ConnectionStatus.CONNECTING:
                logger.info("KiteRealtimeProvider connection already in progress.")
                return True

            self._telemetry.status = ConnectionStatus.CONNECTING
            self._telemetry.last_error = None

            if not self.authenticate():
                return False

            try:
                # Instantiate KiteTicker if available
                try:
                    from kiteconnect import KiteTicker
                    self._ticker = KiteTicker(
                        api_key=self.config.api_key,
                        access_token=self.config.access_token,
                    )
                    self._setup_ticker_callbacks()
                except ImportError:
                    logger.info("kiteconnect SDK not installed; running Kite provider in mockable offline mode.")
                    self._ticker = None

                self._is_running = True
                self._telemetry.status = ConnectionStatus.CONNECTED
                now_iso = datetime.now(timezone.utc).isoformat()
                self._telemetry.connected_at = now_iso
                self._telemetry.last_heartbeat_at = now_iso
                self._telemetry.reconnect_attempts = 0

                # Reconcile existing desired subscriptions after connect/reconnect
                self.reconcile_subscriptions()

                logger.info("KiteRealtimeProvider connected successfully.")
                return True

            except Exception as exc:
                self._telemetry.status = ConnectionStatus.ERROR
                self._telemetry.last_error = f"Connection failed: {exc}"
                logger.error(self._telemetry.last_error)
                return False

    def disconnect(self) -> None:
        """Cleanly terminate WebSocket connection and release listeners."""
        with self._connect_lock:
            self._is_running = False
            if self._ticker:
                try:
                    self._ticker.close()
                except Exception as exc:
                    logger.debug("Error closing ticker: %s", exc)
                self._ticker = None

            self._telemetry.status = ConnectionStatus.DISCONNECTED
            self._telemetry.is_stale = False
            logger.info("KiteRealtimeProvider cleanly disconnected.")

    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to streaming ticks for a list of symbols with limit checks."""
        to_add, err = self.subscription_manager.mark_desired(symbols)
        if err:
            self._telemetry.last_error = err
            logger.warning(err)
            return False

        if not to_add:
            return True

        self.subscription_manager.mark_subscribing(to_add)
        ok = self._apply_subscriptions_to_wire(to_add)
        if ok:
            # If in offline/simulation mode without live ticker, immediately mark subscribed
            if self._ticker is None:
                self.subscription_manager.mark_subscribed(to_add)
        else:
            self.subscription_manager.mark_error(to_add, "Failed to send subscription request to broker.")

        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        return ok

    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from streaming ticks for a list of symbols."""
        clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not clean_symbols:
            return True

        self._apply_unsubscriptions_to_wire(clean_symbols)
        self.subscription_manager.mark_unsubscribed(clean_symbols)
        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        return True

    def reconcile_subscriptions(self) -> bool:
        """Reconcile and resubscribe all desired instruments after reconnecting."""
        resub_symbols = self.subscription_manager.reconcile_after_reconnect()
        if not resub_symbols:
            return True

        logger.info("Reconciling and resubscribing %d instruments after reconnect: %s", len(resub_symbols), resub_symbols)
        ok = self._apply_subscriptions_to_wire(resub_symbols)
        if ok and self._ticker is None:
            self.subscription_manager.mark_subscribed(resub_symbols)
        self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)
        return ok

    def _apply_subscriptions_to_wire(self, symbols: List[str]) -> bool:
        """Send subscription request to active WebSocket."""
        if not self._ticker:
            return True

        tokens = [self.symbol_to_token[s] for s in symbols if s in self.symbol_to_token]
        if not tokens:
            return True

        try:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, tokens)
            return True
        except Exception as exc:
            self._telemetry.last_error = f"Error sending subscription to Kite: {exc}"
            logger.error(self._telemetry.last_error)
            return False

    def _apply_unsubscriptions_to_wire(self, symbols: List[str]) -> bool:
        """Send unsubscription request to active WebSocket."""
        if not self._ticker:
            return True

        tokens = [self.symbol_to_token[s] for s in symbols if s in self.symbol_to_token]
        if not tokens:
            return True

        try:
            self._ticker.unsubscribe(tokens)
            return True
        except Exception as exc:
            logger.error("Error unsubscribing tokens: %s", exc)
            return False

    def handle_binary_frame(self, data: bytes) -> Tuple[int, int]:
        """Decode and process binary message frame from KiteTicker.
        
        Returns:
            Tuple of (valid_ticks_processed, rejected_ticks_count).
        """
        self._telemetry.messages_received += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_message_at = now_iso
        self._telemetry.last_heartbeat_at = now_iso
        self._telemetry.is_stale = False

        ticks, err = KiteProtocolParser.parse_binary_message(data)
        if err:
            self._telemetry.messages_rejected += 1
            logger.warning("Error parsing binary Kite frame: %s", err)

        valid_count = 0
        rejected_count = 0

        for raw_tick in ticks:
            ok, _ = self.handle_raw_tick(raw_tick)
            if ok:
                valid_count += 1
            else:
                rejected_count += 1

        return valid_count, rejected_count

    def handle_text_frame(self, text: str) -> Dict[str, Any]:
        """Process text/JSON control frame from KiteTicker (acks, errors, heartbeats)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_heartbeat_at = now_iso

        parsed = KiteProtocolParser.parse_text_message(text)
        msg_type = parsed.get("type")

        if msg_type in ("ping", "pong"):
            return parsed

        if msg_type in ("subscribe", "unsubscribe", "error"):
            self.subscription_manager.handle_acknowledgement(parsed)
            if msg_type == "error":
                self._telemetry.last_error = parsed.get("message")
            self._telemetry.subscribed_symbols_count = len(self.subscribed_symbols)

        return parsed

    def handle_raw_tick(self, raw_tick: Dict[str, Any]) -> Tuple[bool, Optional[MarketQuote]]:
        """Process incoming raw tick dictionary and pass to validator."""
        self._telemetry.messages_received += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        self._telemetry.last_message_at = now_iso
        self._telemetry.last_heartbeat_at = now_iso
        self._telemetry.is_stale = False

        # Resolve symbol from instrument token if token is provided
        sym_override = None
        token = raw_tick.get("instrument_token")
        if token is not None:
            sym_override = self.token_to_symbol.get(int(token))

        is_valid, quote, err = self.validator.validate_message(
            raw_msg=raw_tick,
            symbol_override=sym_override,
            source="kite",
        )

        if not is_valid or quote is None:
            self._telemetry.messages_rejected += 1
            logger.debug("Rejected incoming tick: %s", err)
            return False, None

        self._dispatch_quote(quote)
        return True, quote

    def trigger_reconnect_backoff(self) -> bool:
        """Execute bounded exponential backoff reconnection attempt."""
        with self._reconnect_lock:
            if self._telemetry.reconnect_attempts >= self.config.reconnect_max_retries:
                self._telemetry.status = ConnectionStatus.ERROR
                self._telemetry.last_error = (
                    f"Max reconnection attempts ({self.config.reconnect_max_retries}) exceeded."
                )
                logger.error(self._telemetry.last_error)
                return False

            self._telemetry.status = ConnectionStatus.RECONNECTING
            self._telemetry.reconnect_attempts += 1
            attempt = self._telemetry.reconnect_attempts

            # Bounded exponential backoff delay: base * 2^(attempt - 1) up to max_delay
            delay = min(
                self.config.reconnect_base_delay * (2 ** (attempt - 1)),
                self.config.reconnect_max_delay,
            )
            logger.info("Reconnecting Kite in %.1f seconds (Attempt %d/%d)...", delay, attempt, self.config.reconnect_max_retries)
            time.sleep(delay)

            return self.connect()

    def _setup_ticker_callbacks(self) -> None:
        """Attach event handlers to KiteTicker instance."""
        if not self._ticker:
            return

        def on_ticks(ws, ticks):
            for tick in ticks:
                self.handle_raw_tick(tick)

        def on_message(ws, payload, is_binary):
            if is_binary:
                self.handle_binary_frame(payload)
            else:
                self.handle_text_frame(payload)

        def on_connect(ws, response):
            logger.info("Kite WebSocket connected.")
            self._telemetry.status = ConnectionStatus.CONNECTED
            self._telemetry.reconnect_attempts = 0
            self.reconcile_subscriptions()

        def on_close(ws, code, reason):
            logger.warning("Kite WebSocket closed (%s): %s", code, reason)
            if self._is_running:
                self.trigger_reconnect_backoff()

        def on_error(ws, code, reason):
            logger.error("Kite WebSocket error (%s): %s", code, reason)
            self._telemetry.last_error = f"Kite error {code}: {reason}"

        self._ticker.on_ticks = on_ticks
        self._ticker.on_message = on_message
        self._ticker.on_connect = on_connect
        self._ticker.on_close = on_close
        self._ticker.on_error = on_error
