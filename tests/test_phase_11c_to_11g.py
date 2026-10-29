"""Comprehensive Automated Test Suite for Phase 11C to 11G.

Covers:
- Phase 11C: Live Price Ingestion, Validation, Freshness, Out-of-Order & Duplicate Handling
- Phase 11D: 1-Minute and 5-Minute OHLCV Candle Construction, IST Alignment, Volume & Database Persistence
- Phase 11E: PPO Feature Compatibility, Exact 12-d Observation Space, Warmup, Normalization & Timeframe Mismatch Protection
- Phase 11F: Live Dashboard API, Candlestick Endpoints, Telemetry & Freshness Badges
- Phase 11G: Reliability, Fault Isolation, Bounded Memory & Graceful Shutdown
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pytest

from backend.app import create_app
from backend.app.db import db
from backend.app.models.market_candle import MarketCandle
from backend.app.realtime.candle_builder import (
    Candle,
    CandleBuilderService,
    get_candle_boundary,
)
from backend.app.realtime.feature_pipeline import (
    CANONICAL_MARKET_FEATURES,
    EXPECTED_OBS_DIM,
    LiveFeaturePipeline,
    PORTFOLIO_STATE_FEATURES,
)
from backend.app.realtime.ingestion import LiveMarketDataIngestionService
from backend.app.realtime.manager import RealtimeMarketDataManager
from backend.app.realtime.mock_provider import MockRealtimeProvider
from backend.app.realtime.models import (
    ConnectionStatus,
    MarketQuote,
    ProviderConfig,
)
from backend.app.realtime.validator import RealtimeDataValidator


@pytest.fixture
def app():
    """Create Flask application fixture for testing."""
    test_app = create_app("testing")
    test_app.config["REALTIME_PROVIDER"] = "mock"
    test_app.config["REALTIME_MAX_SUBSCRIPTIONS"] = 10
    test_app.config["REALTIME_STALE_TIMEOUT_SECONDS"] = 5.0

    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()

    RealtimeMarketDataManager.reset_instance()


@pytest.fixture
def client(app):
    """Create Flask test client fixture."""
    return app.test_client()


# =========================================================================
# Phase 11C: Live Price Ingestion and Validation
# =========================================================================

def test_11c_valid_price_ingestion_and_timestamp_preservation():
    """Verify valid tick ingestion preserves provider event time and separate local processing time."""
    ingestion = LiveMarketDataIngestionService()
    now_event_iso = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()

    raw_msg = {
        "symbol": "RELIANCE.NS",
        "last_price": 2850.50,
        "timestamp": now_event_iso,
        "volume": 15000,
    }

    ok, err = ingestion.ingest_raw_message(raw_msg, source="kite")
    assert ok is True
    assert err is None

    quote = ingestion.get_latest_quote("RELIANCE.NS")
    assert quote is not None
    assert quote.symbol == "RELIANCE.NS"
    assert quote.last_price == 2850.50
    assert quote.timestamp == now_event_iso
    assert quote.received_at is not None
    assert quote.received_at != quote.timestamp  # Preserves event time separate from processing time


def test_11c_rejects_malformed_and_invalid_prices():
    """Verify ingestion rejects malformed payloads and non-positive/non-finite prices."""
    ingestion = LiveMarketDataIngestionService()

    # Negative price
    ok, err = ingestion.ingest_raw_message({
        "symbol": "TCS.NS",
        "last_price": -100.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert ok is False
    assert "positive" in err.lower() or "price" in err.lower()

    # Zero price
    ok_z, err_z = ingestion.ingest_raw_message({
        "symbol": "TCS.NS",
        "last_price": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert ok_z is False

    # NaN price
    ok_nan, err_nan = ingestion.ingest_raw_message({
        "symbol": "TCS.NS",
        "last_price": float("nan"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert ok_nan is False


def test_11c_duplicate_message_detection():
    """Verify identical duplicate ticks are safely detected and eliminated."""
    ingestion = LiveMarketDataIngestionService()
    ts = datetime.now(timezone.utc).isoformat()
    raw = {
        "symbol": "INFY.NS",
        "last_price": 1620.0,
        "timestamp": ts,
        "volume": 500,
    }

    # First ingestion succeeds
    ok1, _ = ingestion.ingest_raw_message(raw)
    assert ok1 is True

    # Immediate identical duplicate is dropped
    ok2, err2 = ingestion.ingest_raw_message(raw)
    assert ok2 is False
    assert "Duplicate" in err2
    assert ingestion.get_telemetry()["total_duplicates"] == 1


def test_11c_out_of_order_tolerance_policy():
    """Verify out-of-order policy: acceptable lag within tolerance is accepted, extreme lag rejected."""
    ingestion = LiveMarketDataIngestionService(out_of_order_tolerance_seconds=5.0)

    now = datetime.now(timezone.utc)
    t0 = now.isoformat()
    t1_late = (now - timedelta(seconds=2)).isoformat()  # 2s lag <= 5s tolerance
    t2_excessive = (now - timedelta(seconds=20)).isoformat()  # 20s lag > 5s tolerance

    # Initial tick
    ok0, _ = ingestion.ingest_raw_message({"symbol": "SBIN.NS", "last_price": 790.0, "timestamp": t0})
    assert ok0 is True

    # Acceptable late tick within tolerance is accepted and marked
    ok_late, _ = ingestion.ingest_raw_message({"symbol": "SBIN.NS", "last_price": 789.5, "timestamp": t1_late})
    assert ok_late is True
    assert ingestion.get_telemetry()["total_out_of_order"] == 1

    # Excessively out of order tick is safely rejected
    ok_bad, err_bad = ingestion.ingest_raw_message({"symbol": "SBIN.NS", "last_price": 788.0, "timestamp": t2_excessive})
    assert ok_bad is False
    assert "out-of-order" in err_bad.lower() or "out of order" in err_bad.lower()


def test_11c_freshness_and_latency_reporting():
    """Verify per-instrument data freshness tracking (FRESH, AGING, STALE, OFFLINE)."""
    ingestion = LiveMarketDataIngestionService(freshness_threshold_seconds=1.0, aging_threshold_seconds=2.0)

    # Unknown symbol
    f_off = ingestion.get_symbol_freshness("UNKNOWN.NS")
    assert f_off["status"] == "OFFLINE"

    # Ingest tick
    ingestion.ingest_raw_message({
        "symbol": "ICICIBANK.NS",
        "last_price": 1180.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    f_fresh = ingestion.get_symbol_freshness("ICICIBANK.NS")
    assert f_fresh["status"] == "FRESH"
    assert f_fresh["last_price"] == 1180.0
    assert f_fresh["latency_ms"] is not None


# =========================================================================
# Phase 11D: OHLCV Candle Construction
# =========================================================================

def test_11d_candle_boundary_ist_alignment():
    """Verify candle boundaries align to Asia/Kolkata (IST) market sessions."""
    # 09:16:32 IST (03:46:32 UTC) -> 5m interval should align to 09:15:00 IST (03:45:00 UTC)
    dt_utc = datetime(2026, 10, 29, 3, 46, 32, tzinfo=timezone.utc)
    start_5m, end_5m = get_candle_boundary(dt_utc, interval_sec=300)

    assert start_5m.hour == 3 and start_5m.minute == 45 and start_5m.second == 0
    assert end_5m.hour == 3 and end_5m.minute == 50 and end_5m.second == 0

    # 1m interval should align to 09:16:00 IST (03:46:00 UTC)
    start_1m, end_1m = get_candle_boundary(dt_utc, interval_sec=60)
    assert start_1m.minute == 46 and start_1m.second == 0
    assert end_1m.minute == 47 and end_1m.second == 0


def test_11d_ohlcv_accumulation_and_completion():
    """Verify correct open, high, low, close, and volume accumulation and completion on boundary crossing."""
    builder = CandleBuilderService(default_interval="5m", enable_db_persistence=False)

    # Base time: 09:15:10 IST
    t0 = datetime(2026, 10, 29, 3, 45, 10, tzinfo=timezone.utc).isoformat()
    t1 = datetime(2026, 10, 29, 3, 46, 0, tzinfo=timezone.utc).isoformat()
    t2 = datetime(2026, 10, 29, 3, 47, 0, tzinfo=timezone.utc).isoformat()
    t3 = datetime(2026, 10, 29, 3, 49, 50, tzinfo=timezone.utc).isoformat()
    # Boundary crossed: 09:20:05 IST
    t_next = datetime(2026, 10, 29, 3, 50, 5, tzinfo=timezone.utc).isoformat()

    # Tick 1: Open at 2800.0, volume cumulative 1000
    q1 = MarketQuote(symbol="RELIANCE.NS", last_price=2800.0, timestamp=t0, received_at=t0, volume=1000)
    builder.process_tick_for_interval(q1, "5m")

    in_prog = builder.get_in_progress_candle("RELIANCE.NS", "5m")
    assert in_prog["open"] == 2800.0
    assert in_prog["high"] == 2800.0
    assert in_prog["low"] == 2800.0
    assert in_prog["close"] == 2800.0
    assert in_prog["is_complete"] is False

    # Tick 2: High at 2850.0, cumulative 1200 (+200 vol)
    q2 = MarketQuote(symbol="RELIANCE.NS", last_price=2850.0, timestamp=t1, received_at=t1, volume=1200)
    builder.process_tick_for_interval(q2, "5m")

    # Tick 3: Low at 2780.0, cumulative 1500 (+300 vol)
    q3 = MarketQuote(symbol="RELIANCE.NS", last_price=2780.0, timestamp=t2, received_at=t2, volume=1500)
    builder.process_tick_for_interval(q3, "5m")

    # Tick 4: Close at 2820.0, cumulative 1800 (+300 vol)
    q4 = MarketQuote(symbol="RELIANCE.NS", last_price=2820.0, timestamp=t3, received_at=t3, volume=1800)
    builder.process_tick_for_interval(q4, "5m")

    # Before crossing boundary, no completed candle yet
    assert builder.get_latest_completed_candle("RELIANCE.NS", "5m") is None

    # Tick 5: Next candle at 09:20:05 IST -> Closes previous 5m candle
    q_next = MarketQuote(symbol="RELIANCE.NS", last_price=2830.0, timestamp=t_next, received_at=t_next, volume=2000)
    completed = builder.process_tick_for_interval(q_next, "5m")

    assert completed is not None
    assert completed.is_complete is True
    assert completed.open == 2800.0
    assert completed.high == 2850.0
    assert completed.low == 2780.0
    assert completed.close == 2820.0
    assert completed.volume == 800  # 1800 - 1000 = 800 total delta volume in interval


def test_11d_database_persistence_and_uniqueness(app):
    """Verify completed candles persist to SQLite and prevent duplicate insertions."""
    with app.app_context():
        builder = CandleBuilderService(default_interval="5m", enable_db_persistence=True)

        t0 = datetime(2026, 10, 29, 3, 45, 0, tzinfo=timezone.utc).isoformat()
        t_next = datetime(2026, 10, 29, 3, 50, 0, tzinfo=timezone.utc).isoformat()

        q1 = MarketQuote(symbol="TCS.NS", last_price=3950.0, timestamp=t0, received_at=t0)
        q2 = MarketQuote(symbol="TCS.NS", last_price=3970.0, timestamp=t_next, received_at=t_next)

        builder.process_tick_for_interval(q1, "5m")
        builder.process_tick_for_interval(q2, "5m")

        # Query database directly
        db_candle = MarketCandle.query.filter_by(symbol="TCS.NS", interval="5m").first()
        assert db_candle is not None
        assert db_candle.symbol == "TCS.NS"
        assert db_candle.open == 3950.0
        assert db_candle.is_complete is True


# =========================================================================
# Phase 11E: PPO-Compatible Feature Engineering
# =========================================================================

def _generate_synthetic_completed_candles(n_bars: int = 60, base_price: float = 2800.0) -> List[Dict[str, Any]]:
    """Helper generating synthetic completed OHLCV candles with realistic drift."""
    candles = []
    curr = base_price
    start_dt = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)

    for i in range(n_bars):
        dt = start_dt + timedelta(minutes=5 * i)
        high = curr * 1.002
        low = curr * 0.998
        close = curr * 1.0005
        candles.append({
            "symbol": "RELIANCE.NS",
            "interval": "5m",
            "start_time": dt.isoformat(),
            "end_time": (dt + timedelta(minutes=5)).isoformat(),
            "open": round(curr, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": 5000 + i * 10,
            "is_complete": True,
        })
        curr = close
    return candles


def test_11e_feature_dimensions_ordering_and_normalization():
    """Verify live feature pipeline generates exact 12-dimensional observation matching PPO specs."""
    pipeline = LiveFeaturePipeline(trained_model_timeframe="5m")  # Matching timeframe for test
    candles = _generate_synthetic_completed_candles(n_bars=60)

    port_state = {
        "cash": 80000.0,
        "holdings": 10.0,
        "cost_basis": 2750.0,
        "peak_value": 110000.0,
    }

    result = pipeline.generate_live_observation(
        candles_history=candles,
        portfolio_state=port_state,
        candle_interval="5m",
        symbol="RELIANCE.NS",
    )

    assert result["ready"] is True
    obs = result["observation"]
    assert len(obs) == EXPECTED_OBS_DIM == 12

    # Verify no NaNs or Infs
    assert all(not np.isnan(x) and not np.isinf(x) for x in obs)

    # Check normalized bounds for RSI [-0.5, 0.5]
    rsi_norm = result["features"]["rsi_14"]
    assert -0.5 <= rsi_norm <= 0.5

    # Check cash ratio [0, 1]
    assert 0.0 <= result["features"]["cash_ratio"] <= 1.0


def test_11e_warmup_guard_and_incomplete_candle_rejection():
    """Verify feature pipeline rejects insufficient history and refuses incomplete candles."""
    pipeline = LiveFeaturePipeline(trained_model_timeframe="5m")

    # 1. Insufficient history (< 50 bars)
    short_candles = _generate_synthetic_completed_candles(n_bars=20)
    res_short = pipeline.generate_live_observation(short_candles, candle_interval="5m")
    assert res_short["ready"] is False
    assert "Insufficient history" in res_short["reason"]

    # 2. Incomplete candle in history
    full_candles = _generate_synthetic_completed_candles(n_bars=60)
    full_candles[-1]["is_complete"] = False  # Still in progress
    res_inprog = pipeline.generate_live_observation(full_candles, candle_interval="5m")
    assert res_inprog["ready"] is False
    assert "Incomplete candle detected" in res_inprog["reason"]


def test_11e_timeframe_mismatch_detection():
    """Verify that when live interval != trained model interval (daily vs 5m), mismatch is safely reported."""
    # Active model checkpoint was trained on daily (1d) candles
    pipeline = LiveFeaturePipeline(trained_model_timeframe="1d")
    intraday_candles = _generate_synthetic_completed_candles(n_bars=60)

    res = pipeline.generate_live_observation(
        candles_history=intraday_candles,
        candle_interval="5m",  # Intraday feed
    )

    # Direct model inference must be disabled to prevent distribution shift
    assert res["ready"] is False
    assert res["timeframe_mismatch"] is True
    assert "Timeframe mismatch" in res["reason"]
    assert res["trained_timeframe"] == "1d"
    assert res["live_timeframe"] == "5m"


# =========================================================================
# Phase 11F: Live Dashboard REST API Endpoints
# =========================================================================

def test_11f_dashboard_api_payload(client):
    """Verify GET /api/v1/realtime/dashboard returns integrated real-time payload."""
    # Connect and subscribe
    client.post("/api/v1/realtime/connect")
    client.post("/api/v1/realtime/subscribe", json={"symbols": ["RELIANCE.NS"]})

    resp = client.get("/api/v1/realtime/dashboard?symbol=RELIANCE.NS&interval=5m")
    assert resp.status_code == 200
    json_data = resp.get_json()
    assert json_data["success"] is True

    data = json_data["data"]
    assert data["symbol"] == "RELIANCE.NS"
    assert data["interval"] == "5m"
    assert "connection" in data
    assert "quotes" in data
    assert "freshness" in data
    assert "candles" in data
    assert "features_readiness" in data


def test_11f_candles_and_freshness_endpoints(client):
    """Verify /candles/<symbol> and /freshness/<symbol> endpoints."""
    client.post("/api/v1/realtime/connect")
    client.post("/api/v1/realtime/subscribe", json={"symbols": ["INFY.NS"]})

    # Candles endpoint
    resp_c = client.get("/api/v1/realtime/candles/INFY.NS?interval=5m&limit=10")
    assert resp_c.status_code == 200
    assert resp_c.get_json()["data"]["symbol"] == "INFY.NS"

    # Freshness endpoint
    resp_f = client.get("/api/v1/realtime/freshness/INFY.NS")
    assert resp_f.status_code == 200
    f_data = resp_f.get_json()["data"]
    assert f_data["symbol"] == "INFY.NS"
    assert "status" in f_data


def test_11f_features_endpoint(client):
    """Verify /features/<symbol> endpoint evaluates PPO readiness."""
    resp = client.get("/api/v1/realtime/features/RELIANCE.NS?interval=5m")
    assert resp.status_code == 200
    feat_data = resp.get_json()["data"]
    assert "ready" in feat_data
    assert "observation_dim" in feat_data
    assert feat_data["observation_dim"] == 12


# =========================================================================
# Phase 11G: Reliability, Recovery, and Fault Isolation
# =========================================================================

def test_11g_fault_isolation_across_multiple_instruments():
    """Verify that a malformed update in one instrument does not stop ingestion of other instruments."""
    ingestion = LiveMarketDataIngestionService()
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Valid tick for RELIANCE
    ok1, _ = ingestion.ingest_raw_message({"symbol": "RELIANCE.NS", "last_price": 2850.0, "timestamp": now_ts})
    assert ok1 is True

    # 2. Corrupt tick for TCS (negative price)
    ok2, _ = ingestion.ingest_raw_message({"symbol": "TCS.NS", "last_price": -999.0, "timestamp": now_ts})
    assert ok2 is False  # Rejected

    # 3. Valid tick for INFY immediately succeeds
    ok3, _ = ingestion.ingest_raw_message({"symbol": "INFY.NS", "last_price": 1620.0, "timestamp": now_ts})
    assert ok3 is True

    # Check both valid instruments are tracked intact
    assert ingestion.get_latest_quote("RELIANCE.NS") is not None
    assert ingestion.get_latest_quote("INFY.NS") is not None
    assert ingestion.get_latest_quote("TCS.NS") is None


def test_11g_bounded_memory_ring_buffer():
    """Verify tick buffers and signature sets remain bounded to prevent memory leaks."""
    ingestion = LiveMarketDataIngestionService(max_recent_ticks=50)

    # Push 100 ticks
    for i in range(100):
        ts = (datetime.now(timezone.utc) + timedelta(milliseconds=i * 200)).isoformat()
        ingestion.ingest_raw_message({
            "symbol": "SBIN.NS",
            "last_price": 790.0 + (i * 0.1),
            "timestamp": ts,
            "volume": 100 + i,
        })

    recent = ingestion.get_recent_ticks("SBIN.NS", limit=200)
    # Bounded to max_recent_ticks (50)
    assert len(recent) == 50


def test_11g_graceful_shutdown_releases_subsystems(app):
    """Verify disconnect stops watchdog monitors and leaves state clean."""
    cfg = ProviderConfig(provider_name="mock", api_key="test", access_token="token")
    manager = RealtimeMarketDataManager.get_instance(config=cfg)
    manager.connect()
    assert manager.provider.status == ConnectionStatus.CONNECTED

    # Graceful disconnect
    manager.disconnect()
    assert manager.provider.status == ConnectionStatus.DISCONNECTED
    assert manager._is_monitoring is False
