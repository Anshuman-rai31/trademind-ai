"""Unit and Integration Tests for Real-Time Market Data Provider Integration (Phase 11A).

Tests:
    1. Provider Abstraction & Contracts (Kite, Mock, Base).
    2. Authentication:
       - Missing credentials -> AUTH_FAILED.
       - Valid credentials -> CONNECTED.
       - Credential isolation (zero token leakage in logs/API).
    3. Real-Time Message Validation:
       - Valid quote parsing (LTP, OHLC, depth, timestamps).
       - Rejection of malformed messages (non-dict, non-numeric price, missing symbol).
       - Negative or zero price rejection.
       - Future clock-skew rejection (> 60s).
       - Stale event timestamp rejection (> 300s).
       - Duplicate tick rejection.
       - Out-of-order timestamp rejection.
    4. Connection Reliability & Reconnection:
       - Graceful disconnect.
       - Bounded exponential backoff reconnection.
       - Feed staleness detection (updates stopped arriving).
    5. Portfolio-Aware Subscriptions:
       - Add/remove symbols without reconnecting.
       - Subscription limits enforcement.
       - Synchronization with active portfolio stock bucket.
    6. Flask REST API Integration:
       - GET /api/v1/realtime/status (no secrets leaked).
       - POST /api/v1/realtime/connect.
       - GET & POST /api/v1/realtime/subscriptions, /subscribe, /unsubscribe.
       - POST /api/v1/realtime/sync-portfolio.
       - GET /api/v1/realtime/quotes and /quotes/<symbol>.
"""
import time
from datetime import datetime, timedelta, timezone
import pytest

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
from backend.app.realtime.validator import RealtimeDataValidator
from backend.app.services.trading_pipeline_service import TradingPipelineService


@pytest.fixture(autouse=True)
def reset_realtime_manager():
    """Reset RealtimeMarketDataManager singleton before each test."""
    RealtimeMarketDataManager.reset_instance()
    yield
    RealtimeMarketDataManager.reset_instance()


# ==============================================================================
# 1. Validation Layer Tests
# ==============================================================================

def test_validator_valid_quote():
    """Test validation and normalization of a valid market tick."""
    validator = RealtimeDataValidator()
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_tick = {
        "symbol": "RELIANCE.NS",
        "last_price": 2850.50,
        "timestamp": now_iso,
        "volume": 125000,
        "open": 2840.0,
        "high": 2865.0,
        "low": 2835.0,
        "close": 2845.0,
        "bid": 2850.0,
        "ask": 2851.0,
    }

    ok, quote, err = validator.validate_message(raw_tick)
    assert ok is True
    assert err is None
    assert isinstance(quote, MarketQuote)
    assert quote.symbol == "RELIANCE.NS"
    assert quote.last_price == 2850.50
    assert quote.volume == 125000
    assert quote.bid_price == 2850.0
    assert quote.ask_price == 2851.0
    assert quote.timestamp == now_iso
    assert quote.received_at is not None


def test_validator_token_mapping():
    """Test mapping raw numeric instrument token to canonical NSE symbol."""
    validator = RealtimeDataValidator()
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_tick = {
        "instrument_token": 738561,  # RELIANCE
        "last_price": 2900.0,
        "timestamp": now_iso,
    }

    ok, quote, err = validator.validate_message(raw_tick, symbol_override="RELIANCE.NS")
    assert ok is True
    assert quote.symbol == "RELIANCE.NS"


def test_validator_rejects_malformed_message():
    """Test non-dict or corrupt payloads are rejected."""
    validator = RealtimeDataValidator()
    ok, quote, err = validator.validate_message("corrupt_string")
    assert ok is False
    assert quote is None
    assert "Malformed message" in err


def test_validator_rejects_missing_symbol():
    """Test message without symbol is rejected."""
    validator = RealtimeDataValidator()
    ok, quote, err = validator.validate_message({"last_price": 100.0, "timestamp": datetime.now(timezone.utc).isoformat()})
    assert ok is False
    assert "missing a valid instrument identifier" in err


def test_validator_rejects_unsupported_symbol():
    """Test message for unsupported asset is rejected."""
    validator = RealtimeDataValidator()
    ok, quote, err = validator.validate_message({
        "symbol": "BITCOIN.CRYPTO",
        "last_price": 50000.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert ok is False
    assert "not among supported instruments" in err


def test_validator_rejects_negative_or_nan_price():
    """Test negative, zero, NaN, or non-numeric prices are rejected."""
    validator = RealtimeDataValidator()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Zero price
    ok, _, err = validator.validate_message({"symbol": "TCS.NS", "last_price": 0.0, "timestamp": now_iso})
    assert ok is False
    assert "strictly positive" in err

    # Negative price
    ok, _, err = validator.validate_message({"symbol": "TCS.NS", "last_price": -50.0, "timestamp": now_iso})
    assert ok is False
    assert "strictly positive" in err

    # NaN price
    ok, _, err = validator.validate_message({"symbol": "TCS.NS", "last_price": float("nan"), "timestamp": now_iso})
    assert ok is False
    assert "cannot be NaN" in err

    # Non-numeric string
    ok, _, err = validator.validate_message({"symbol": "TCS.NS", "last_price": "invalid", "timestamp": now_iso})
    assert ok is False
    assert "must be numeric" in err


def test_validator_rejects_clock_skew_future_timestamp():
    """Test message with timestamp too far into the future is rejected."""
    validator = RealtimeDataValidator(max_clock_skew_seconds=10.0)
    future_ts = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
    ok, _, err = validator.validate_message({"symbol": "INFY.NS", "last_price": 1600.0, "timestamp": future_ts})
    assert ok is False
    assert "Clock skew error" in err


def test_validator_rejects_stale_event_timestamp():
    """Test message with excessively old timestamp is rejected."""
    validator = RealtimeDataValidator(max_staleness_seconds=60.0)
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
    ok, _, err = validator.validate_message({"symbol": "INFY.NS", "last_price": 1600.0, "timestamp": stale_ts})
    assert ok is False
    assert "Stale message" in err


def test_validator_rejects_duplicate_tick():
    """Test duplicate consecutive tick is eliminated."""
    validator = RealtimeDataValidator()
    now_iso = datetime.now(timezone.utc).isoformat()
    tick = {"symbol": "SBIN.NS", "last_price": 750.0, "timestamp": now_iso}

    ok1, quote1, _ = validator.validate_message(tick)
    assert ok1 is True

    # Immediate duplicate with same timestamp and price
    ok2, quote2, err2 = validator.validate_message(tick)
    assert ok2 is False
    assert "Duplicate tick detected" in err2


def test_validator_rejects_out_of_order_tick():
    """Test tick arriving with older timestamp than previously ingested tick is rejected."""
    validator = RealtimeDataValidator()
    now_dt = datetime.now(timezone.utc)
    t1 = now_dt.isoformat()
    t0 = (now_dt - timedelta(seconds=5)).isoformat()

    validator.validate_message({"symbol": "ICICIBANK.NS", "last_price": 1150.0, "timestamp": t1})
    ok, _, err = validator.validate_message({"symbol": "ICICIBANK.NS", "last_price": 1151.0, "timestamp": t0})
    assert ok is False
    assert "Out-of-order tick" in err


# ==============================================================================
# 2. Provider Abstraction & Authentication Tests
# ==============================================================================

def test_kite_provider_missing_credentials_auth_failure():
    """Test that Kite provider fails authentication cleanly when credentials are missing."""
    config = ProviderConfig(api_key=None, access_token=None)
    provider = KiteRealtimeProvider(config=config)
    assert provider.authenticate() is False
    assert provider.status == ConnectionStatus.AUTH_FAILED
    assert "Missing KITE_API_KEY" in provider.telemetry.last_error


def test_kite_provider_safe_config_no_leakage():
    """Test that provider credentials are never leaked in serialization."""
    config = ProviderConfig(
        provider_name="kite",
        api_key="super_secret_key_12345",
        access_token="super_secret_token_abcde",
    )
    safe = config.to_safe_dict()
    assert "api_key" not in safe
    assert "access_token" not in safe
    assert safe["has_credentials"] is True
    assert safe["has_api_key"] is True
    assert safe["has_access_token"] is True


def test_kite_provider_connect_success_with_credentials():
    """Test connecting Kite provider when credentials format is satisfied."""
    config = ProviderConfig(
        api_key="valid_api_key",
        access_token="valid_access_token_length_ok",
    )
    provider = KiteRealtimeProvider(config=config)
    connected = provider.connect()
    assert connected is True
    assert provider.status == ConnectionStatus.CONNECTED
    assert provider.telemetry.connected_at is not None

    provider.disconnect()
    assert provider.status == ConnectionStatus.DISCONNECTED


def test_mock_provider_lifecycle():
    """Test mock provider connect, tick generation, and disconnect."""
    provider = MockRealtimeProvider()
    assert provider.connect() is True
    assert provider.status == ConnectionStatus.CONNECTED

    # Subscribe symbols
    provider.subscribe(["RELIANCE.NS", "TCS.NS"])
    assert provider.subscribed_symbols == ["RELIANCE.NS", "TCS.NS"]

    # Emit ticks
    quotes = provider.emit_mock_ticks_for_subscribed()
    assert len(quotes) == 2
    assert provider.get_latest_quote("RELIANCE.NS") is not None
    assert provider.get_latest_quote("TCS.NS") is not None

    # Unsubscribe
    provider.unsubscribe(["TCS.NS"])
    assert provider.subscribed_symbols == ["RELIANCE.NS"]

    provider.disconnect()
    assert provider.status == ConnectionStatus.DISCONNECTED


def test_mock_provider_simulated_auth_failure():
    """Test mock provider error injection for authentication failure."""
    provider = MockRealtimeProvider(simulate_auth_failure=True)
    assert provider.connect() is False
    assert provider.status == ConnectionStatus.AUTH_FAILED


# ==============================================================================
# 3. Connection Reliability & Reconnection Tests
# ==============================================================================

def test_kite_provider_bounded_exponential_backoff(monkeypatch):
    """Test bounded exponential backoff on reconnection failures."""
    config = ProviderConfig(
        api_key="key_ok",
        access_token="token_ok_length_sufficient",
        reconnect_base_delay=0.01,  # Fast for test execution
        reconnect_max_delay=0.05,
        reconnect_max_retries=3,
    )
    provider = KiteRealtimeProvider(config=config)
    # Simulate connection failure to test backoff retry counter
    monkeypatch.setattr(provider, "connect", lambda: False)

    # Trigger reconnects
    provider.trigger_reconnect_backoff()
    assert provider.telemetry.reconnect_attempts == 1

    provider.trigger_reconnect_backoff()
    assert provider.telemetry.reconnect_attempts == 2

    provider.trigger_reconnect_backoff()
    assert provider.telemetry.reconnect_attempts == 3

    # 4th exceeds max_retries
    ok = provider.trigger_reconnect_backoff()
    assert ok is False
    assert provider.status == ConnectionStatus.ERROR
    assert "exceeded" in provider.telemetry.last_error


def test_staleness_detection():
    """Test that connection is flagged as STALE if updates stop arriving."""
    config = ProviderConfig(stale_timeout_seconds=0.1)
    provider = MockRealtimeProvider(config=config)
    manager = RealtimeMarketDataManager(provider=provider, config=config)

    manager.connect()
    manager.subscribe(["RELIANCE.NS"])
    provider.emit_mock_ticks_for_subscribed()

    assert manager.get_status() == ConnectionStatus.CONNECTED

    # Wait beyond stale threshold
    time.sleep(0.2)
    assert manager.check_staleness() is True
    assert manager.get_status() == ConnectionStatus.STALE
    assert manager.get_telemetry().is_stale is True

    # Sending a fresh tick clears staleness
    provider.emit_mock_ticks_for_subscribed()
    assert manager.get_telemetry().is_stale is False


# ==============================================================================
# 4. Portfolio-Aware Subscriptions Tests
# ==============================================================================

def test_portfolio_aware_subscription_sync(app):
    """Test synchronizing real-time subscriptions with a portfolio stock bucket."""
    with app.app_context():
        pipeline_service = TradingPipelineService()
        port = pipeline_service.create_portfolio(
            name="Realtime_Portfolio",
            initial_cash=100000.0,
            bucket=[{"symbol": "RELIANCE.NS"}, {"symbol": "INFY.NS"}],
        )

        provider = MockRealtimeProvider()
        manager = RealtimeMarketDataManager(provider=provider)
        manager.connect()

        # Initial sync
        bucket_syms = [b.symbol for b in port.bucket_items]
        res = manager.sync_portfolio_subscriptions(bucket_syms, port.name)

        assert res["success"] is True
        assert res["subscribed_symbols"] == ["INFY.NS", "RELIANCE.NS"]
        assert res["added"] == ["INFY.NS", "RELIANCE.NS"]
        assert res["removed"] == []

        # Update portfolio bucket: add TCS.NS, remove INFY.NS
        pipeline_service.update_portfolio_bucket("Realtime_Portfolio", [
            {"symbol": "RELIANCE.NS"},
            {"symbol": "TCS.NS"},
        ])
        updated_port = pipeline_service.get_portfolio("Realtime_Portfolio")
        new_bucket_syms = [b.symbol for b in updated_port.bucket_items]

        res2 = manager.sync_portfolio_subscriptions(new_bucket_syms, updated_port.name)
        assert res2["success"] is True
        assert res2["subscribed_symbols"] == ["RELIANCE.NS", "TCS.NS"]
        assert res2["added"] == ["TCS.NS"]
        assert res2["removed"] == ["INFY.NS"]


def test_subscription_limits_enforced():
    """Test exceeding max_subscriptions limit is rejected."""
    config = ProviderConfig(max_subscriptions=2)
    provider = MockRealtimeProvider(config=config)
    provider.connect()

    assert provider.subscribe(["RELIANCE.NS", "TCS.NS"]) is True
    # 3rd symbol exceeds limit
    assert provider.subscribe(["INFY.NS"]) is False
    assert len(provider.subscribed_symbols) == 2


# ==============================================================================
# 5. REST API Integration Tests
# ==============================================================================

def test_api_realtime_status_no_secrets(client):
    """Test GET /api/v1/realtime/status returns valid state without secrets."""
    res = client.get("/api/v1/realtime/status")
    assert res.status_code == 200
    data = res.get_json()["data"]

    assert "status" in data
    assert "provider" in data
    assert "telemetry" in data
    assert "config" in data

    # Verify security: no raw credentials in response
    assert "api_key" not in data["config"]
    assert "access_token" not in data["config"]


def test_api_realtime_subscribe_and_quotes(client):
    """Test subscribing and retrieving quotes via REST API."""
    # Connect
    res_conn = client.post("/api/v1/realtime/connect")
    # In test mode without KITE credentials, it gracefully reports status
    assert res_conn.status_code in (200, 503)

    # Use mock provider in manager to test full API cycle
    manager = RealtimeMarketDataManager.get_instance()
    mock_prov = MockRealtimeProvider()
    mock_prov.connect()
    manager.set_provider(mock_prov)

    # Subscribe
    res_sub = client.post("/api/v1/realtime/subscribe", json={"symbols": ["RELIANCE.NS", "SBIN.NS"]})
    assert res_sub.status_code == 200
    assert "RELIANCE.NS" in res_sub.get_json()["data"]["current_subscriptions"]

    # Generate a tick
    mock_prov.emit_mock_ticks_for_subscribed()

    # Get all quotes
    res_quotes = client.get("/api/v1/realtime/quotes")
    assert res_quotes.status_code == 200
    quotes_data = res_quotes.get_json()["data"]
    assert quotes_data["count"] == 2
    assert "RELIANCE.NS" in quotes_data["quotes"]

    # Get single quote
    res_single = client.get("/api/v1/realtime/quotes/RELIANCE.NS")
    assert res_single.status_code == 200
    q = res_single.get_json()["data"]
    assert q["symbol"] == "RELIANCE.NS"
    assert q["last_price"] > 0

    # Unsubscribe
    res_unsub = client.post("/api/v1/realtime/unsubscribe", json={"symbols": ["SBIN.NS"]})
    assert res_unsub.status_code == 200
    assert "SBIN.NS" not in res_unsub.get_json()["data"]["current_subscriptions"]
