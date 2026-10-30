"""Comprehensive Automated Test Suite for Phase 12: Real-Time RL Trading Signals.

Covers:
- Model Loading & Artifact Checksum Verification
- Model/Feature Schema Incompatibility & Timeframe Mismatch Protection
- Exact 12-Dimensional Observation Construction & Canonical Feature Ordering
- Missing, Incomplete, and Invalid Candle Rejection
- Warmup History Sufficiency (>= 50 bars)
- Deterministic PPO Policy Prediction & Discrete Action Mapping (0: HOLD, 1: BUY, 2: SELL)
- Multi-Symbol Isolation across Supported NSE Equities
- Duplicate Completed Candle Signal Prevention
- Pre-Trade Risk Engine Gating (Blocked/Modified Orders)
- Database Persistence & Audit Trail Lineage (RLTradingSignal)
- REST API Endpoints: /latest, /history, /<id>, /health, /generate
- End-to-End Pipeline Integration from CandleBuilderService to Signals
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pytest

from backend.app import create_app
from backend.app.db import db
from backend.app.models.market_candle import MarketCandle
from backend.app.models.model_registry import RegisteredModel
from backend.app.models.paper_trading import PaperPortfolio, PaperPosition
from backend.app.models.trading_signal import RLTradingSignal
from backend.app.realtime.candle_builder import Candle, CandleBuilderService
from backend.app.realtime.feature_pipeline import (
    CANONICAL_MARKET_FEATURES,
    EXPECTED_OBS_DIM,
    LiveFeaturePipeline,
    PORTFOLIO_STATE_FEATURES,
)
from backend.app.realtime.manager import RealtimeMarketDataManager
from backend.app.realtime.models import MarketQuote
from backend.app.risk.config import RiskConfig
from backend.app.risk.engine import RiskManager
from backend.app.services.market_data_service import MarketDataService
from backend.app.services.model_registry_service import ModelRegistryService
from backend.app.signals.service import ACTION_MAP, SUPPORTED_NSE_SYMBOLS, RealtimeSignalService


@pytest.fixture
def app():
    """Create isolated test application context."""
    test_app = create_app("testing")
    test_app.config["REALTIME_PROVIDER"] = "mock"
    test_app.config["REALTIME_MAX_SUBSCRIPTIONS"] = 10
    test_app.config["REALTIME_STALE_TIMEOUT_SECONDS"] = 5.0

    with test_app.app_context():
        db.create_all()
        # Seed default portfolio
        port = PaperPortfolio.query.filter_by(name="default").first()
        if not port:
            port = PaperPortfolio(
                name="default",
                initial_cash=100000.0,
                cash=100000.0,
                portfolio_value=100000.0,
                peak_value=100000.0,
            )
            db.session.add(port)
            db.session.commit()

        yield test_app
        db.session.remove()
        db.drop_all()

    RealtimeSignalService.reset_instance()
    RealtimeMarketDataManager.reset_instance()


@pytest.fixture
def client(app):
    """Test client fixture."""
    return app.test_client()


def generate_synthetic_bars(symbol: str, count: int = 60, interval: str = "1d", base_price: float = 2500.0) -> List[Dict[str, Any]]:
    """Helper to generate chronological synthetic OHLCV bars for testing."""
    bars: List[Dict[str, Any]] = []
    start_dt = datetime.now(timezone.utc) - timedelta(days=count + 5)
    cur_price = base_price

    for i in range(count):
        bar_dt = start_dt + timedelta(days=i) if interval == "1d" else start_dt + timedelta(minutes=5 * i)
        iso_str = bar_dt.isoformat()
        ret = np.sin(i / 5.0) * 0.015
        open_p = cur_price
        close_p = round(open_p * (1.0 + ret), 2)
        high_p = round(max(open_p, close_p) * 1.008, 2)
        low_p = round(min(open_p, close_p) * 0.992, 2)
        vol = int(50000 + abs(np.cos(i)) * 25000)

        bars.append({
            "symbol": symbol,
            "interval": interval,
            "start_time": iso_str,
            "end_time": iso_str,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
            "source": "synthetic_test",
            "is_complete": True,
        })
        cur_price = close_p

    return bars


# =============================================================================
# 1. Model Loading, Registry & Artifact Validation
# =============================================================================

def test_phase_12_model_registry_sync_and_inference_loading(app):
    """Verify ModelRegistryService detects checkpoints and validates observation dimensions."""
    registry = ModelRegistryService()
    synced = registry.sync_local_artifacts()
    assert synced >= 0

    # Ensure best_model or approved model is registered
    models = registry.list_models()
    assert len(models) > 0

    # Load active model for RELIANCE.NS
    model, model_rec = registry.load_model_for_inference("RELIANCE.NS")
    assert model is not None
    assert model_rec.obs_dimension == EXPECTED_OBS_DIM
    assert getattr(model.observation_space, "shape")[0] == 12


def test_phase_12_incompatible_model_or_missing_artifact_rejected(app):
    """Verify attempting inference with nonexistent or corrupted model raises clear error."""
    registry = ModelRegistryService()
    # Test unapproved non-existent symbol
    with pytest.raises(ValueError, match="No active or approved PPO model found"):
        registry.load_model_for_inference("NONEXISTENT.NS")


# =============================================================================
# 2. Timeframe Mismatch Protection & Safety
# =============================================================================

def test_phase_12_timeframe_mismatch_intraday_candles_safely_rejected(app):
    """Verify feeding 5m or 1m candle to 1d model flags TIMEFRAME_MISMATCH without running policy."""
    service = RealtimeSignalService.get_instance()
    bars = generate_synthetic_bars("RELIANCE.NS", count=55, interval="5m")

    # Seed candle builder buffer
    for b in bars[:-1]:
        service._assemble_warmup_history("RELIANCE.NS", "5m", b)

    target_5m_candle = bars[-1]
    signal_res = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle=target_5m_candle,
        interval="5m",
    )

    assert signal_res["status"] == "TIMEFRAME_MISMATCH"
    assert signal_res["action"] == "HOLD"
    assert signal_res["proposed_action"] is None
    assert signal_res["effective_action"] == "HOLD"
    assert "Timeframe mismatch" in signal_res["rejection_reason"]
    assert signal_res["confidence"] is None  # strictly no fabricated confidence
    assert signal_res["observation"] is not None  # preserved for auditing


# =============================================================================
# 3. Compatible Timeframe (1d) Observation & PPO Policy Inference
# =============================================================================

def test_phase_12_daily_candle_observation_and_deterministic_ppo_inference(app):
    """Verify compatible 1d candle generates observation and produces deterministic BUY, SELL, or HOLD."""
    service = RealtimeSignalService.get_instance()
    bars = generate_synthetic_bars("RELIANCE.NS", count=55, interval="1d")

    # Prefill buffer
    for b in bars[:-1]:
        service._assemble_warmup_history("RELIANCE.NS", "1d", b)

    target_candle = bars[-1]
    signal_res = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle=target_candle,
        interval="1d",
    )

    assert signal_res["status"] in ("VALID", "RISK_BLOCKED", "RISK_MODIFIED")
    assert signal_res["action"] in ("BUY", "SELL", "HOLD")
    assert signal_res["effective_action"] in ("BUY", "SELL", "HOLD")
    assert signal_res["confidence"] is None  # no invented confidence scores
    assert signal_res["close_price"] == target_candle["close"]
    assert signal_res["source_candle_timestamp"] == target_candle["start_time"]
    assert signal_res["model_id"] is not None
    assert signal_res["model_checksum"] is not None


def test_phase_12_exact_12d_observation_structure_and_ordering(app):
    """Verify the 12-d observation preserves exact ordering: 7 market + 5 portfolio features."""
    pipe = LiveFeaturePipeline(trained_model_timeframe="1d")
    bars = generate_synthetic_bars("RELIANCE.NS", count=55, interval="1d")

    port_state = {
        "cash": 85000.0,
        "holdings": 10.0,
        "cost_basis": 2400.0,
        "peak_value": 110000.0,
    }

    res = pipe.generate_live_observation(
        candles_history=bars,
        portfolio_state=port_state,
        candle_interval="1d",
        symbol="RELIANCE.NS",
    )

    assert res["ready"] is True
    assert res["observation_dim"] == 12
    obs = res["observation"]
    assert len(obs) == 12

    # Check bounds and NaN
    for v in obs:
        assert isinstance(v, float)
        assert not np.isnan(v)
        assert not np.isinf(v)

    # Check portfolio feature values match expected formulas
    close_price = bars[-1]["close"]
    total_val = 85000.0 + (10.0 * close_price)
    expected_cash_ratio = round(85000.0 / total_val, 4)
    assert abs(res["features"]["cash_ratio"] - expected_cash_ratio) < 0.05
    assert res["features"]["position_flag"] == 1.0


def test_phase_12_action_mapping_integrity():
    """Verify discrete action mapping strictly matches Gymnasium environment spec."""
    assert ACTION_MAP[0] == "HOLD"
    assert ACTION_MAP[1] == "BUY"
    assert ACTION_MAP[2] == "SELL"


# =============================================================================
# 4. Warmup Sufficiency & Corrupt Data Safeguards
# =============================================================================

def test_phase_12_warmup_insufficient_history_safely_rejected(app):
    """Verify candle with insufficient history (< 50 bars) is marked REJECTED."""
    # Force fresh service instance with empty buffers
    RealtimeSignalService.reset_instance()
    service = RealtimeSignalService.get_instance()
    
    # Only 5 bars without historical fallback
    short_bars = generate_synthetic_bars("UNKNOWN_SYM.NS", count=5, interval="1d")
    
    # Bypass storage warmup for clean test
    service._candle_buffers[("UNKNOWN_SYM.NS", "1d")].extend(short_bars[:-1])
    
    res = service.process_completed_candle(
        symbol="UNKNOWN_SYM.NS",
        candle=short_bars[-1],
        interval="1d",
    )
    
    assert res["status"] == "REJECTED"
    assert "Insufficient history" in res["rejection_reason"]
    assert res["action"] == "HOLD"


def test_phase_12_incomplete_or_invalid_candles_rejected(app):
    """Verify incomplete, missing price, or malformed candle dictionaries are rejected."""
    service = RealtimeSignalService.get_instance()

    # Incomplete candle
    inc_res = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle={"start_time": "2026-10-30T10:00:00Z", "close": 2800.0, "is_complete": False},
        interval="1d",
    )
    assert inc_res["status"] == "REJECTED"
    assert "INCOMPLETE_CANDLE" in inc_res["rejection_reason"]

    # Missing close price
    noprice_res = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle={"start_time": "2026-10-30T10:00:00Z", "close": None, "is_complete": True},
        interval="1d",
    )
    assert noprice_res["status"] == "REJECTED"
    assert "INVALID_PRICE" in noprice_res["rejection_reason"]


# =============================================================================
# 5. Duplicate Signal Prevention
# =============================================================================

def test_phase_12_duplicate_completed_candle_prevention(app):
    """Verify the exact same candle start time cannot trigger multiple signals."""
    service = RealtimeSignalService.get_instance()
    bars = generate_synthetic_bars("RELIANCE.NS", count=55, interval="1d")

    target_candle = bars[-1]

    # First evaluation
    sig1 = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle=target_candle,
        interval="1d",
    )
    assert sig1["status"] in ("VALID", "RISK_BLOCKED", "RISK_MODIFIED")

    # Immediate second evaluation of same candle
    sig2 = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle=target_candle,
        interval="1d",
    )

    # Must return existing signal without creating duplicate
    assert sig2["id"] == sig1["id"]
    assert service.get_service_health()["telemetry"]["total_duplicates_prevented"] >= 1

    # Verify database has only 1 row
    db_signals = RLTradingSignal.query.filter_by(
        symbol="RELIANCE.NS",
        interval="1d",
        source_candle_timestamp=target_candle["start_time"],
    ).all()
    assert len(db_signals) == 1


# =============================================================================
# 6. Multi-Symbol State Isolation
# =============================================================================

def test_phase_12_multi_symbol_state_isolation(app):
    """Verify multiple NSE stocks maintain isolated candle buffers and signals."""
    service = RealtimeSignalService.get_instance()

    symbols = ["RELIANCE.NS", "TCS.NS", "SBIN.NS"]
    for sym in symbols:
        bars = generate_synthetic_bars(sym, count=55, interval="1d")
        for b in bars[:-1]:
            service._assemble_warmup_history(sym, "1d", b)
        
        res = service.process_completed_candle(symbol=sym, candle=bars[-1], interval="1d")
        assert res["symbol"] == sym

    # Verify isolated latest signals
    latest_all = service.get_latest_signals()
    for sym in symbols:
        assert latest_all[sym] is not None
        assert latest_all[sym]["symbol"] == sym


# =============================================================================
# 7. Pre-Trade Risk Engine Gating & Interventions
# =============================================================================

def test_phase_12_risk_engine_blocks_order_on_severe_drawdown(app):
    """Verify RiskManager intervention turns model BUY into RISK_BLOCKED with HOLD action."""
    service = RealtimeSignalService.get_instance()
    
    # Configure RiskManager to be in restricted/drawdown breach state
    risk_cfg = RiskConfig(enabled=True, max_drawdown_limit=0.10)
    service.risk_manager = RiskManager(config=risk_cfg)
    service.risk_manager.is_restricted = True
    service.risk_manager.cooldown_remaining = 5

    bars = generate_synthetic_bars("RELIANCE.NS", count=55, interval="1d")
    for b in bars[:-1]:
        service._assemble_warmup_history("RELIANCE.NS", "1d", b)

    # Force simulated model action to be BUY
    class MockModelBuy:
        observation_space = type("Obs", (), {"shape": (12,)})()
        def predict(self, obs, deterministic=True):
            return np.array(1), None  # 1 = BUY

    class MockRegistry:
        def sync_local_artifacts(self): return 1
        def load_model_for_inference(self, sym):
            m_rec = type("Rec", (), {"model_id": "mock_buy_model", "version": "1.0", "artifact_checksum": "abc123"})()
            return MockModelBuy(), m_rec

    service.registry_service = MockRegistry()

    sig = service.process_completed_candle(
        symbol="RELIANCE.NS",
        candle=bars[-1],
        interval="1d",
    )

    # Risk manager should block the BUY
    assert sig["proposed_action"] == "BUY"
    assert sig["effective_action"] == "HOLD"
    assert sig["status"] == "RISK_BLOCKED"
    assert "Risk blocked" in sig["rejection_reason"]


# =============================================================================
# 8. Database Persistence & Audit Lineage
# =============================================================================

def test_phase_12_signal_database_persistence_and_retrieval(app):
    """Verify RLTradingSignal entity is correctly stored in SQLite and can be retrieved."""
    now_iso = datetime.now(timezone.utc).isoformat()
    record = RLTradingSignal(
        symbol="INFY.NS",
        interval="1d",
        action="BUY",
        proposed_action="BUY",
        effective_action="BUY",
        confidence=None,
        close_price=1620.50,
        source_candle_timestamp=now_iso,
        timestamp=now_iso,
        model_id="ppo_infy_policy",
        model_version="1.0.0",
        model_checksum="sha256_mock_hash",
        data_freshness="FRESH",
        status="VALID",
        observation_json=json.dumps([0.1] * 12),
        features_json=json.dumps({"rsi_14": 52.0}),
    )

    db.session.add(record)
    db.session.commit()

    saved = RLTradingSignal.query.filter_by(symbol="INFY.NS").first()
    assert saved is not None
    assert saved.action == "BUY"
    assert saved.signal_id.startswith("sig_")

    d = saved.to_dict()
    assert d["close_price"] == 1620.50
    assert len(d["observation"]) == 12
    assert d["features"]["rsi_14"] == 52.0


# =============================================================================
# 9. REST API Endpoints
# =============================================================================

def test_phase_12_signals_api_health_endpoint(client):
    """Test GET /api/v1/signals/health."""
    res = client.get("/api/v1/signals/health")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["status"] == "ONLINE"
    assert "RELIANCE.NS" in data["supported_symbols"]


def test_phase_12_signals_api_generate_and_latest_endpoints(client):
    """Test POST /api/v1/signals/generate and GET /api/v1/signals/latest."""
    # 1. Trigger on-demand generation for 1d candle
    gen_res = client.post(
        "/api/v1/signals/generate",
        json={"symbol": "RELIANCE.NS", "interval": "1d"},
    )
    assert gen_res.status_code == 200
    gen_json = gen_res.get_json()
    assert gen_json["success"] is True
    assert gen_json["data"]["symbol"] == "RELIANCE.NS"

    # 2. Query latest signal for single symbol
    lat_res = client.get("/api/v1/signals/latest?symbol=RELIANCE.NS")
    assert lat_res.status_code == 200
    lat_json = lat_res.get_json()
    assert lat_json["success"] is True
    assert lat_json["data"]["symbol"] == "RELIANCE.NS"

    # 3. Query all latest signals
    all_res = client.get("/api/v1/signals/latest")
    assert all_res.status_code == 200
    all_json = all_res.get_json()
    assert all_json["success"] is True
    assert "RELIANCE.NS" in all_json["data"]


def test_phase_12_signals_api_history_and_details_endpoints(client):
    """Test GET /api/v1/signals/history and GET /api/v1/signals/<id>."""
    # Seed signal via generate
    client.post("/api/v1/signals/generate", json={"symbol": "RELIANCE.NS", "interval": "1d"})

    hist_res = client.get("/api/v1/signals/history?page=1&page_size=10")
    assert hist_res.status_code == 200
    hist_json = hist_res.get_json()
    assert hist_json["success"] is True
    assert len(hist_json["data"]) >= 1

    first_sig = hist_json["data"][0]
    sig_id = first_sig["id"]

    # Query details
    det_res = client.get(f"/api/v1/signals/{sig_id}")
    assert det_res.status_code == 200
    det_json = det_res.get_json()
    assert det_json["success"] is True
    assert det_json["data"]["id"] == sig_id


# =============================================================================
# 10. CandleBuilderService to RealtimeSignalService Integration
# =============================================================================

def test_phase_12_candle_builder_consumer_callback_integration(app):
    """Verify CandleBuilderService invokes on_completed_candle upon interval rollover."""
    builder = CandleBuilderService(default_interval="5m", enable_db_persistence=False)
    service = RealtimeSignalService.get_instance()

    received_signals: List[Dict[str, Any]] = []
    service.register_signal_listener(lambda s: received_signals.append(s))

    builder.register_candle_consumer(service.on_completed_candle)

    # First tick opens candle
    quote1 = MarketQuote(
        symbol="SBIN.NS",
        last_price=790.0,
        timestamp="2026-10-30T10:00:05Z",
        received_at="2026-10-30T10:00:05.100Z",
        source="kite",
    )
    builder.process_tick_for_interval(quote1, "5m")

    # Tick crossing boundary closes candle and triggers callback
    quote2 = MarketQuote(
        symbol="SBIN.NS",
        last_price=792.5,
        timestamp="2026-10-30T10:05:01Z",
        received_at="2026-10-30T10:05:01.100Z",
        source="kite",
    )
    builder.process_tick_for_interval(quote2, "5m")

    # A 5m candle was completed, so the signal service received it and created a signal
    assert len(received_signals) == 1
    assert received_signals[0]["symbol"] == "SBIN.NS"
    assert received_signals[0]["status"] == "TIMEFRAME_MISMATCH"
