"""Unit and Integration Tests for Phase 11B: WebSocket Connection and Subscription Management.

Tests:
- Connection lifecycle: connecting, connected, disconnecting, duplicate connection prevention
- Authentication failure: missing/malformed credentials
- Subscription management: add, remove, duplicate prevention, limits enforcement
- Subscription acknowledgements and broker error responses
- Reconnection with bounded exponential backoff
- Automatic resubscription and reconciliation after reconnecting
- Binary packet parsing: LTP (8B), Quote (44B), Full (184B) modes
- Malformed message resilience: truncated packets, corrupted headers, bad JSON
- Distinguishing inactive market instruments from broken connection feeds
- REST APIs: /health, /reconnect, /subscriptions?details=true
"""
from __future__ import annotations

import json
import struct
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest

from backend.app import create_app
from backend.app.realtime.base_provider import RealtimeDataProvider
from backend.app.realtime.kite_provider import (
    DEFAULT_KITE_NSE_TOKEN_MAP,
    KiteRealtimeProvider,
)
from backend.app.realtime.manager import RealtimeMarketDataManager
from backend.app.realtime.mock_provider import MockRealtimeProvider
from backend.app.realtime.models import (
    ConnectionStatus,
    ConnectionTelemetry,
    MarketQuote,
    ProviderConfig,
)
from backend.app.realtime.parser import KiteProtocolParser
from backend.app.realtime.subscription import (
    InstrumentActivityStatus,
    SubscriptionManager,
    SubscriptionState,
)
from backend.app.realtime.validator import RealtimeDataValidator


@pytest.fixture
def app():
    """Create Flask application fixture for testing."""
    test_app = create_app("testing")
    test_app.config["REALTIME_PROVIDER"] = "mock"
    test_app.config["REALTIME_MAX_SUBSCRIPTIONS"] = 10
    test_app.config["REALTIME_STALE_TIMEOUT_SECONDS"] = 5.0
    yield test_app
    RealtimeMarketDataManager.reset_instance()


@pytest.fixture
def client(app):
    """Create Flask test client fixture."""
    return app.test_client()


# =========================================================================
# 1. Connection Lifecycle and Duplicate Connection Prevention
# =========================================================================

def test_connection_lifecycle_and_prevent_duplicate_connect():
    """Verify clean state progression and prevention of duplicate concurrent connections."""
    cfg = ProviderConfig(provider_name="kite", api_key="test_key", access_token="test_access_token_12345")
    provider = KiteRealtimeProvider(config=cfg)

    assert provider.status == ConnectionStatus.DISCONNECTED

    # First connect
    ok = provider.connect()
    assert ok is True
    assert provider.status == ConnectionStatus.CONNECTED
    assert provider.telemetry.connected_at is not None
    assert provider.telemetry.reconnect_attempts == 0

    # Duplicate connect should safely return True without recreating connection
    ok_dup = provider.connect()
    assert ok_dup is True
    assert provider.status == ConnectionStatus.CONNECTED

    # Clean disconnect
    provider.disconnect()
    assert provider.status == ConnectionStatus.DISCONNECTED


def test_auth_failure_with_missing_and_malformed_credentials():
    """Verify auth failure handling when credentials are missing or malformed."""
    # Missing credentials
    cfg_empty = ProviderConfig(provider_name="kite", api_key=None, access_token=None)
    provider_empty = KiteRealtimeProvider(config=cfg_empty)
    assert provider_empty.connect() is False
    assert provider_empty.status == ConnectionStatus.AUTH_FAILED
    assert "Missing" in provider_empty.telemetry.last_error

    # Malformed credentials (too short)
    cfg_short = ProviderConfig(provider_name="kite", api_key="a", access_token="b")
    provider_short = KiteRealtimeProvider(config=cfg_short)
    assert provider_short.connect() is False
    assert provider_short.status == ConnectionStatus.AUTH_FAILED


# =========================================================================
# 2. Subscription Management & State Tracking
# =========================================================================

def test_subscription_state_transitions():
    """Verify transitions: DESIRED -> SUBSCRIBING -> SUBSCRIBED -> UNSUBSCRIBED."""
    tokens = {"RELIANCE.NS": 738561, "TCS.NS": 2953217}
    sub_mgr = SubscriptionManager(symbol_to_token=tokens, max_subscriptions=5)

    assert sub_mgr.get_subscribed_symbols() == []

    # Mark desired
    to_add, err = sub_mgr.mark_desired(["RELIANCE.NS"])
    assert err is None
    assert to_add == ["RELIANCE.NS"]
    assert sub_mgr.get_desired_symbols() == ["RELIANCE.NS"]

    # Mark subscribing
    sub_mgr.mark_subscribing(to_add)
    assert sub_mgr.get_subscribed_symbols() == ["RELIANCE.NS"]

    # Mark subscribed
    sub_mgr.mark_subscribed(to_add)
    assert sub_mgr.get_subscribed_symbols() == ["RELIANCE.NS"]

    details = sub_mgr.get_subscription_details()
    assert details["instruments"]["RELIANCE.NS"]["state"] == SubscriptionState.SUBSCRIBED.value

    # Mark unsubscribed
    sub_mgr.mark_unsubscribed(["RELIANCE.NS"])
    assert sub_mgr.get_subscribed_symbols() == []
    assert sub_mgr.get_desired_symbols() == []


def test_subscription_limits_and_duplicate_prevention():
    """Verify subscription capacity limits and deduplication of requests."""
    tokens = {f"SYM{i}.NS": i for i in range(1, 10)}
    sub_mgr = SubscriptionManager(symbol_to_token=tokens, max_subscriptions=2)

    to_add, err = sub_mgr.mark_desired(["SYM1.NS", "SYM2.NS"])
    assert err is None
    assert to_add == ["SYM1.NS", "SYM2.NS"]
    sub_mgr.mark_subscribed(to_add)

    # Adding a 3rd should fail capacity check
    to_add_extra, err_extra = sub_mgr.mark_desired(["SYM3.NS"])
    assert err_extra is not None
    assert "limit exceeded" in err_extra

    # Re-requesting already subscribed symbol should return empty to_add without error
    to_add_dup, err_dup = sub_mgr.mark_desired(["SYM1.NS"])
    assert err_dup is None
    assert to_add_dup == []


def test_subscription_acknowledgements_and_broker_errors():
    """Verify processing of broker JSON control frames for acks and errors."""
    tokens = {"RELIANCE.NS": 738561, "TCS.NS": 2953217}
    cfg = ProviderConfig(provider_name="kite", api_key="test_key", access_token="test_access_token_12345")
    provider = KiteRealtimeProvider(config=cfg, token_map=tokens)
    provider.connect()

    provider.subscribe(["RELIANCE.NS", "TCS.NS"])

    # Simulate broker subscription acknowledgement frame
    ack_frame = json.dumps({"type": "subscribe", "status": True, "data": [738561, 2953217]})
    resp = provider.handle_text_frame(ack_frame)
    assert resp["type"] == "subscribe"
    assert resp["status"] is True

    details = provider.get_subscription_details()
    assert details["instruments"]["RELIANCE.NS"]["state"] == SubscriptionState.SUBSCRIBED.value
    assert details["instruments"]["TCS.NS"]["state"] == SubscriptionState.SUBSCRIBED.value

    # Simulate unsubscription ack
    unsub_frame = json.dumps({"type": "unsubscribe", "status": True, "data": [2953217]})
    provider.handle_text_frame(unsub_frame)
    assert "TCS.NS" not in provider.subscribed_symbols

    # Simulate broker error frame
    err_frame = json.dumps({"type": "error", "message": "Instrument not available for trading"})
    provider.handle_text_frame(err_frame)
    assert provider.telemetry.last_error == "Instrument not available for trading"


# =========================================================================
# 3. Reconnection and Resubscription Reconciliation
# =========================================================================

def test_reconnection_reconciles_subscriptions(monkeypatch):
    """Verify that subscriptions are reconciled and resent after reconnection."""
    tokens = {"RELIANCE.NS": 738561, "TCS.NS": 2953217}
    cfg = ProviderConfig(
        provider_name="kite",
        api_key="test_key",
        access_token="test_access_token_12345",
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
    )
    provider = KiteRealtimeProvider(config=cfg, token_map=tokens)
    provider.connect()

    # Subscribe to RELIANCE and TCS
    provider.subscribe(["RELIANCE.NS", "TCS.NS"])
    assert sorted(provider.subscribed_symbols) == ["RELIANCE.NS", "TCS.NS"]

    # Track calls to _apply_subscriptions_to_wire
    applied_symbols = []
    original_apply = provider._apply_subscriptions_to_wire

    def mock_apply(symbols):
        applied_symbols.extend(symbols)
        return original_apply(symbols)

    monkeypatch.setattr(provider, "_apply_subscriptions_to_wire", mock_apply)

    # Disconnect
    provider.disconnect()
    assert provider.status == ConnectionStatus.DISCONNECTED

    # Reconnect - should automatically reconcile and resubscribe desired symbols
    provider.connect()
    assert provider.status == ConnectionStatus.CONNECTED
    assert "RELIANCE.NS" in applied_symbols
    assert "TCS.NS" in applied_symbols
    assert sorted(provider.subscribed_symbols) == ["RELIANCE.NS", "TCS.NS"]


# =========================================================================
# 4. Binary and Control Frame Protocol Parsing
# =========================================================================

def test_parser_ltp_binary_frame():
    """Verify parsing of KiteTicker 8-byte LTP binary packets."""
    raw_frame = KiteProtocolParser.build_binary_packet(
        instrument_token=738561,
        mode="ltp",
        last_price=2850.50,
    )
    ticks, err = KiteProtocolParser.parse_binary_message(raw_frame)
    assert err is None
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["instrument_token"] == 738561
    assert tick["mode"] == "ltp"
    assert tick["last_price"] == 2850.50


def test_parser_quote_binary_frame():
    """Verify parsing of KiteTicker 44-byte Quote binary packets."""
    raw_frame = KiteProtocolParser.build_binary_packet(
        instrument_token=2953217,
        mode="quote",
        last_price=3950.00,
        volume=125000,
        open_price=3920.00,
        high_price=3965.50,
        low_price=3910.00,
        close_price=3940.00,
    )
    ticks, err = KiteProtocolParser.parse_binary_message(raw_frame)
    assert err is None
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["instrument_token"] == 2953217
    assert tick["mode"] == "quote"
    assert tick["last_price"] == 3950.00
    assert tick["volume"] == 125000
    assert tick["open_price"] == 3920.00
    assert tick["high_price"] == 3965.50
    assert tick["low_price"] == 3910.00
    assert tick["close_price"] == 3940.00


def test_parser_full_binary_frame_with_depth_and_exchange_timestamp():
    """Verify parsing of KiteTicker 184-byte Full packets with depth and exchange timestamp."""
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    raw_frame = KiteProtocolParser.build_binary_packet(
        instrument_token=408065,
        mode="full",
        last_price=1620.25,
        volume=54000,
        exchange_timestamp=now_epoch,
        best_bid_price=1620.00,
        best_bid_qty=200,
        best_ask_price=1620.50,
        best_ask_qty=350,
    )
    ticks, err = KiteProtocolParser.parse_binary_message(raw_frame)
    assert err is None
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["instrument_token"] == 408065
    assert tick["mode"] == "full"
    assert tick["last_price"] == 1620.25
    assert tick["bid"] == 1620.00
    assert tick["bid_quantity"] == 200
    assert tick["ask"] == 1620.50
    assert tick["ask_quantity"] == 350
    assert "depth" in tick
    assert len(tick["depth"]["buy"]) == 5
    assert len(tick["depth"]["sell"]) == 5
    assert "exchange_timestamp" in tick


def test_parser_malformed_and_truncated_binary_packets():
    """Verify that corrupt or truncated binary frames are handled safely without crashing."""
    # Truncated header (< 2 bytes)
    ticks1, err1 = KiteProtocolParser.parse_binary_message(b"\x00")
    assert len(ticks1) == 0
    assert "too short" in err1

    # Header says 1 packet of length 44, but only 10 bytes provided
    bad_frame = struct.pack(">HH", 1, 44) + b"\x00" * 10
    ticks2, err2 = KiteProtocolParser.parse_binary_message(bad_frame)
    assert len(ticks2) == 0
    assert "truncated" in err2.lower()

    # Empty payload
    ticks3, err3 = KiteProtocolParser.parse_binary_message(b"")
    assert len(ticks3) == 0
    assert "Empty" in err3


def test_parser_text_control_frames():
    """Verify handling of text control frames: ping, pong, json, and bad json."""
    p1 = KiteProtocolParser.parse_text_message("ping")
    assert p1["type"] == "ping"

    p2 = KiteProtocolParser.parse_text_message("pong")
    assert p2["type"] == "pong"

    p3 = KiteProtocolParser.parse_text_message('{"type": "subscribe", "status": true, "data": [123]}')
    assert p3["type"] == "subscribe"
    assert p3["data"] == [123]

    p4 = KiteProtocolParser.parse_text_message("NOT A VALID JSON STRING")
    assert p4["type"] == "text"
    assert p4["content"] == "NOT A VALID JSON STRING"


# =========================================================================
# 5. Reliability: Inactive Market Instruments vs Broken Connection
# =========================================================================

def test_distinguish_inactive_instrument_from_broken_connection():
    """Verify distinguishing between an inactive stock (low volume) and a dead socket."""
    tokens = {"RELIANCE.NS": 738561, "TCS.NS": 2953217}
    sub_mgr = SubscriptionManager(
        symbol_to_token=tokens,
        inactive_threshold_seconds=1.0,
    )
    sub_mgr.mark_desired(["RELIANCE.NS", "TCS.NS"])
    sub_mgr.mark_subscribed(["RELIANCE.NS", "TCS.NS"])

    # Simulate tick received for RELIANCE 2 seconds ago
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    sub_mgr.record_tick("RELIANCE.NS", 2850.0, old_time)

    # Case A: Connection is ALIVE (heartbeat is good) -> RELIANCE is INACTIVE_MARKET (illiquid/off-hours)
    sub_mgr.evaluate_instrument_activity(is_connection_alive=True)
    details_alive = sub_mgr.get_subscription_details()
    assert details_alive["instruments"]["RELIANCE.NS"]["activity_status"] == InstrumentActivityStatus.INACTIVE_MARKET.value

    # Case B: Connection is DEAD (socket dropped / no heartbeat) -> RELIANCE is STALE
    sub_mgr.evaluate_instrument_activity(is_connection_alive=False)
    details_dead = sub_mgr.get_subscription_details()
    assert details_dead["instruments"]["RELIANCE.NS"]["activity_status"] == InstrumentActivityStatus.STALE.value


# =========================================================================
# 6. REST API Endpoints: Health, Reconnect, and Detailed Subscriptions
# =========================================================================

def test_api_realtime_health_endpoint(client):
    """Verify GET /api/v1/realtime/health returns granular diagnostic status."""
    resp = client.get("/api/v1/realtime/health")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert "status" in data
    assert "connected" in data
    assert "is_stale" in data
    assert "subscriptions" in data
    assert "active_instruments" in data["subscriptions"]
    assert "inactive_market_instruments" in data["subscriptions"]


def test_api_realtime_reconnect_endpoint(client):
    """Verify POST /api/v1/realtime/reconnect cleans up and reconciles subscriptions."""
    # First connect and subscribe to an instrument
    client.post("/api/v1/realtime/connect")
    client.post("/api/v1/realtime/subscribe", json={"symbols": ["RELIANCE.NS"]})

    # Trigger reconnect
    resp = client.post("/api/v1/realtime/reconnect")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["reconnected"] is True
    assert "RELIANCE.NS" in data["subscriptions"]


def test_api_realtime_subscriptions_details_query(client):
    """Verify GET /api/v1/realtime/subscriptions?details=true returns instrument breakdown."""
    client.post("/api/v1/realtime/connect")
    client.post("/api/v1/realtime/subscribe", json={"symbols": ["INFY.NS"]})

    resp = client.get("/api/v1/realtime/subscriptions?details=true")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert "instruments" in data
    assert "INFY.NS" in data["instruments"]
    assert data["instruments"]["INFY.NS"]["state"] == SubscriptionState.SUBSCRIBED.value
