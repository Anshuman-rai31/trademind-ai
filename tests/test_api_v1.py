"""Comprehensive Automated Test Suite for Phase 9 Flask REST API & SQLite Integration.

Tests cover:
    1. Health & System Status Endpoints
    2. Asset Management & Validated Historical Data
    3. Dashboard Platform Summary
    4. Backtesting Engine Integration, Async/Sync Execution, Trades & Equity Curves
    5. PPO Agent Status, Metrics, and Model Decision Inference
    6. Risk Management Gateway & Order Evaluation
    7. Market-Regime Information & Empirical Cluster Summaries
    8. Input Validation, Bounds Checking & Error Envelopes
    9. Database Integrity, Foreign Keys & Cascading Deletions
"""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from backend.app import create_app
from backend.app.db import db
from backend.app.models.agent import AgentRun
from backend.app.models.asset import Asset
from backend.app.models.backtest import BacktestRun, PortfolioSnapshotRecord, TradeRecord
from backend.app.regime.config import RegimeConfig
from backend.app.regime.detector import MarketRegimeDetector


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def test_app():
    """Create testing Flask application with in-memory SQLite database."""
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    """Test HTTP client."""
    return test_app.test_client()


# ==============================================================================
# 1. Health & System Status Tests
# ==============================================================================

def test_api_v1_health_endpoint(client):
    """Test GET /api/v1/health returns 200 with database connectivity and UTC timestamp."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.is_json

    body = response.get_json()
    assert body["success"] is True
    assert "data" in body
    assert "meta" in body
    assert "timestamp" in body["meta"]

    data = body["data"]
    assert data["status"] == "healthy"
    assert data["app_name"] == "TradeMind"
    assert data["database"]["connected"] is True
    assert data["database"]["engine"] == "sqlite"


def test_api_v1_health_degraded_when_db_down(client):
    """Test GET /api/v1/health returns 503 when database execution fails."""
    with patch("backend.app.db.db.session.execute", side_effect=Exception("Database connection error")):
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        body = response.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_api_v1_system_status(client):
    """Test GET /api/v1/system/status returns modular capabilities summary."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True

    data = body["data"]
    assert "market_data" in data
    assert data["market_data"]["market"] == "NSE"
    assert "baseline_strategies" in data
    strategy_ids = [s["id"] for s in data["baseline_strategies"]]
    assert "buy_and_hold" in strategy_ids
    assert "ma_crossover" in strategy_ids

    assert "ppo_agent" in data
    assert data["ppo_agent"]["training_support_installed"] is True
    assert "saved_model_artifact_available" in data["ppo_agent"]
    assert "ready_for_inference" in data["ppo_agent"]

    assert "risk_management" in data
    assert data["risk_management"]["status"] == "operational"

    assert "market_regime" in data
    assert data["market_regime"]["installed"] is True


# ==============================================================================
# 2. Asset Management Tests
# ==============================================================================

def test_api_v1_list_assets(client):
    """Test GET /api/v1/assets returns seeded canonical assets."""
    response = client.get("/api/v1/assets")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True

    assets = body["data"]["assets"]
    assert len(assets) >= 5
    symbols = [a["symbol"] for a in assets]
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols
    assert "INFY.NS" in symbols


def test_api_v1_get_asset_details(client):
    """Test GET /api/v1/assets/<symbol> for existing and missing assets."""
    # Existing asset
    res_ok = client.get("/api/v1/assets/RELIANCE.NS")
    assert res_ok.status_code == 200
    body_ok = res_ok.get_json()
    assert body_ok["success"] is True
    assert body_ok["data"]["symbol"] == "RELIANCE.NS"
    assert body_ok["data"]["exchange"] == "NSE"

    # Non-existent asset
    res_404 = client.get("/api/v1/assets/UNKNOWN.NS")
    assert res_404.status_code == 404
    body_404 = res_404.get_json()
    assert body_404["success"] is False
    assert body_404["error"]["code"] == "ASSET_NOT_FOUND"


def test_api_v1_asset_history_success(client):
    """Test GET /api/v1/assets/<symbol>/history returns validated OHLCV bars."""
    response = client.get("/api/v1/assets/RELIANCE.NS/history?limit=5")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    bars = body["data"]
    assert len(bars) <= 5
    assert len(bars) > 0

    first_bar = bars[0]
    for required in ("timestamp", "open", "high", "low", "close", "volume", "symbol"):
        assert required in first_bar
    assert first_bar["symbol"] == "RELIANCE.NS"


def test_api_v1_asset_history_date_filtering(client):
    """Test date-range filtering on historical bars."""
    response = client.get("/api/v1/assets/RELIANCE.NS/history?start_date=2024-01-02&end_date=2024-01-08")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    for bar in body["data"]:
        ts_date = bar["timestamp"].split("T")[0]
        assert "2024-01-01" <= ts_date <= "2024-01-09"


def test_api_v1_asset_history_validation_errors(client):
    """Test error handling for invalid date ranges, limits, and unavailable symbols."""
    # 1. Invalid start_date format
    r1 = client.get("/api/v1/assets/RELIANCE.NS/history?start_date=01-01-2024")
    assert r1.status_code == 400
    assert r1.get_json()["error"]["code"] == "INVALID_DATE_FORMAT"

    # 2. start_date > end_date
    r2 = client.get("/api/v1/assets/RELIANCE.NS/history?start_date=2024-01-10&end_date=2024-01-05")
    assert r2.status_code == 400
    assert r2.get_json()["error"]["code"] == "INVALID_DATE_RANGE"

    # 3. Invalid limit
    r3 = client.get("/api/v1/assets/RELIANCE.NS/history?limit=-5")
    assert r3.status_code == 400
    assert r3.get_json()["error"]["code"] == "INVALID_PARAMETER"

    # 4. Symbol without processed historical data
    r4 = client.get("/api/v1/assets/SBIN.NS/history")
    assert r4.status_code == 404
    assert r4.get_json()["error"]["code"] == "DATA_UNAVAILABLE"


# ==============================================================================
# 3. Dashboard Summary Tests
# ==============================================================================

def test_api_v1_dashboard_summary(client):
    """Test GET /api/v1/dashboard/summary reflects persisted state without fake data."""
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]

    assert data["supported_assets_count"] >= 5
    assert data["completed_backtests_count"] == 0
    assert data["latest_completed_backtest"] is None
    assert data["latest_portfolio_valuation"] is None
    assert "ppo_agent" in data
    assert "risk_configuration" in data


# ==============================================================================
# 4. Backtest Engine Tests
# ==============================================================================

def test_api_v1_create_backtest_buy_and_hold(client):
    """Test creating and running a Buy and Hold backtest synchronously."""
    payload = {
        "name": "Reliance Buy and Hold Test",
        "symbol": "RELIANCE.NS",
        "strategy_type": "buy_and_hold",
        "start_date": "2023-12-31",
        "end_date": "2024-01-11",
        "initial_capital": 100000.0,
        "parameters": {"allocation_pct": 0.95},
    }

    # In testing mode, sync=true runs simulation immediately and writes results to SQLite
    response = client.post("/api/v1/backtests?sync=true", json=payload)
    assert response.status_code in (200, 201)
    body = response.get_json()
    assert body["success"] is True

    run_data = body["data"]
    assert run_data["name"] == "Reliance Buy and Hold Test"
    assert run_data["symbol"] == "RELIANCE.NS"
    assert run_data["strategy_type"] == "buy_and_hold"
    assert run_data["status"] == "completed"
    assert run_data["metrics"] is not None
    assert "total_return" in run_data["metrics"]

    backtest_id = run_data["id"]

    # Verify GET /api/v1/backtests/<id>
    get_res = client.get(f"/api/v1/backtests/{backtest_id}")
    assert get_res.status_code == 200
    assert get_res.get_json()["data"]["id"] == backtest_id

    # Verify GET /api/v1/backtests/<id>/trades
    trades_res = client.get(f"/api/v1/backtests/{backtest_id}/trades")
    assert trades_res.status_code == 200
    trades_body = trades_res.get_json()
    assert trades_body["success"] is True
    assert len(trades_body["data"]) >= 1
    assert trades_body["data"][0]["action"] == "BUY"

    # Verify GET /api/v1/backtests/<id>/equity-curve
    curve_res = client.get(f"/api/v1/backtests/{backtest_id}/equity-curve")
    assert curve_res.status_code == 200
    curve_body = curve_res.get_json()
    assert curve_body["success"] is True
    assert len(curve_body["data"]) > 0
    first_snap = curve_body["data"][0]
    assert "cash" in first_snap
    assert "portfolio_value" in first_snap

    # Verify GET /api/v1/backtests/<id>/metrics
    metrics_res = client.get(f"/api/v1/backtests/{backtest_id}/metrics")
    assert metrics_res.status_code == 200
    assert "sharpe_ratio" in metrics_res.get_json()["data"]


def test_api_v1_create_backtest_ma_crossover(client):
    """Test creating a Moving Average Crossover backtest."""
    payload = {
        "name": "Reliance MA Crossover Test",
        "symbol": "RELIANCE.NS",
        "strategy_type": "ma_crossover",
        "start_date": "2023-12-31",
        "end_date": "2024-01-11",
        "initial_capital": 50000.0,
        "parameters": {
            "fast_window": 2,
            "slow_window": 5,
            "allocation_pct": 0.90,
        },
    }

    response = client.post("/api/v1/backtests?sync=true", json=payload)
    assert response.status_code in (200, 201)
    body = response.get_json()
    assert body["data"]["status"] == "completed"
    assert body["data"]["initial_capital"] == 50000.0


def test_api_v1_backtest_validation_errors(client):
    """Test input validation and error responses for backtest requests."""
    # 1. Malformed JSON
    r1 = client.post("/api/v1/backtests", data="invalid json string", content_type="application/json")
    assert r1.status_code == 400
    assert r1.get_json()["error"]["code"] == "MALFORMED_JSON"

    # 2. Missing name
    r2 = client.post("/api/v1/backtests", json={"symbol": "RELIANCE.NS", "strategy_type": "buy_and_hold"})
    assert r2.status_code == 400

    # 3. Unsupported strategy
    r3 = client.post("/api/v1/backtests", json={"name": "Test", "symbol": "RELIANCE.NS", "strategy_type": "arbitrary_ai"})
    assert r3.status_code == 400
    assert r3.get_json()["error"]["code"] == "UNSUPPORTED_STRATEGY"

    # 4. Negative initial capital
    r4 = client.post("/api/v1/backtests", json={
        "name": "Test",
        "symbol": "RELIANCE.NS",
        "strategy_type": "buy_and_hold",
        "initial_capital": -1000.0,
    })
    assert r4.status_code == 400

    # 5. Invalid symbol
    r5 = client.post("/api/v1/backtests", json={
        "name": "Test",
        "symbol": "INVALID$$SYMBOL",
        "strategy_type": "buy_and_hold",
    })
    assert r5.status_code == 400
    assert r5.get_json()["error"]["code"] == "INVALID_SYMBOL"

    # 6. Non-existent backtest ID
    r6 = client.get("/api/v1/backtests/99999")
    assert r6.status_code == 404
    assert r6.get_json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_api_v1_list_backtests_pagination_and_filters(client):
    """Test listing backtest runs with pagination and status filters."""
    # Create two backtests
    client.post("/api/v1/backtests?sync=true", json={
        "name": "Run 1",
        "symbol": "RELIANCE.NS",
        "strategy_type": "buy_and_hold",
    })
    client.post("/api/v1/backtests?sync=true", json={
        "name": "Run 2",
        "symbol": "RELIANCE.NS",
        "strategy_type": "buy_and_hold",
    })

    # Test listing
    res = client.get("/api/v1/backtests?limit=1&page=1")
    assert res.status_code == 200
    body = res.get_json()
    assert len(body["data"]) == 1
    assert body["meta"]["total"] >= 2
    assert body["meta"]["pages"] >= 2

    # Filter by symbol
    res_sym = client.get("/api/v1/backtests?symbol=RELIANCE.NS")
    assert res_sym.status_code == 200
    assert len(res_sym.get_json()["data"]) >= 2


# ==============================================================================
# 5. PPO Agent Tests
# ==============================================================================

def test_api_v1_agent_status_without_artifact(client):
    """Test GET /api/v1/agent/status correctly reports artifact availability."""
    response = client.get("/api/v1/agent/status")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]

    assert data["training_support_installed"] is True
    assert data["algorithm"] == "PPO"
    assert "saved_model_artifact_available" in data
    assert "ready_for_inference" in data


def test_api_v1_agent_metrics_without_runs(client):
    """Test GET /api/v1/agent/metrics handles empty evaluation status safely."""
    response = client.get("/api/v1/agent/metrics")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["data"] is None or isinstance(body["data"], dict)


def test_api_v1_agent_decision_missing_model_returns_503(client):
    """Test POST /api/v1/agent/decisions returns 503 SERVICE_UNAVAILABLE when no artifact exists."""
    payload = {
        "symbol": "RELIANCE.NS",
        "state": [1.0, 0.5, -0.2, 0.8, 1.2],
    }
    response = client.post("/api/v1/agent/decisions", json=payload)
    # If no model is saved in models/, endpoint MUST return 503 rather than inventing decisions
    assert response.status_code == 503
    body = response.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"


def test_api_v1_agent_decision_with_mock_model(client):
    """Test POST /api/v1/agent/decisions with a loaded mock PPO policy."""
    class MockModel:
        observation_space = type("Space", (), {"shape": (5,)})()
        def predict(self, obs, deterministic=True):
            return 1, None  # BUY

    with patch("backend.app.services.agent_service.AgentService.find_model_artifact", return_value=type("Path", (), {"exists": lambda self: True, "name": "best_model.zip"})()):
        with patch("backend.app.services.agent_service.AgentService._load_model", return_value=MockModel()):
            payload = {
                "symbol": "RELIANCE.NS",
                "state": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
            response = client.post("/api/v1/agent/decisions", json=payload)
            assert response.status_code == 200
            body = response.get_json()
            assert body["success"] is True
            data = body["data"]
            assert data["symbol"] == "RELIANCE.NS"
            assert data["action"] == 1
            assert data["action_name"] == "BUY"
            assert "research and simulation purposes only" in data["disclaimer"]


# ==============================================================================
# 6. Risk Management Tests
# ==============================================================================

def test_api_v1_risk_config_endpoint(client):
    """Test GET /api/v1/risk/config returns active thresholds."""
    response = client.get("/api/v1/risk/config")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]
    assert data["enabled"] is True
    assert "max_position_exposure" in data
    assert "max_drawdown_limit" in data


def test_api_v1_risk_status_endpoint(client):
    """Test GET /api/v1/risk/status returns operational readiness."""
    response = client.get("/api/v1/risk/status")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]
    assert data["status"] == "operational"
    assert data["live_portfolio_active"] is False


def test_api_v1_risk_evaluate_approved_order(client):
    """Test POST /api/v1/risk/evaluate approves compliant order."""
    payload = {
        "symbol": "RELIANCE.NS",
        "side": "BUY",
        "quantity": 10.0,
        "price": 1280.0,
        "portfolio": {
            "cash": 100000.0,
            "holdings": {},
        },
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]
    assert data["approved"] is True
    assert data["intervention_occurred"] is False
    assert data["effective_action"]["quantity"] == 10.0


def test_api_v1_risk_evaluate_intervention(client):
    """Test POST /api/v1/risk/evaluate intervenes when order exceeds capital."""
    payload = {
        "symbol": "RELIANCE.NS",
        "side": "BUY",
        "quantity": 1000.0,
        "price": 1280.0,  # 1.28M exceeds 100k cash
        "portfolio": {
            "cash": 100000.0,
            "holdings": {},
        },
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    data = body["data"]
    # Risk manager rejects unaffordable order (exceeds cash)
    assert data["intervention_occurred"] is True
    assert data["approved"] is False
    assert data["effective_action"] is None
    assert data["intervention_reason"] == "INSUFFICIENT_CASH"


def test_api_v1_risk_evaluate_validation_errors(client):
    """Test error handling on invalid order side, negative quantity, or price."""
    # Invalid side
    r1 = client.post("/api/v1/risk/evaluate", json={"symbol": "RELIANCE.NS", "side": "HOLD", "quantity": 10, "price": 100})
    assert r1.status_code == 400

    # Negative quantity
    r2 = client.post("/api/v1/risk/evaluate", json={"symbol": "RELIANCE.NS", "side": "BUY", "quantity": -5, "price": 100})
    assert r2.status_code == 400


# ==============================================================================
# 7. Market-Regime Tests
# ==============================================================================

def test_api_v1_regime_endpoints_when_model_missing(client):
    """Test regime endpoints return 404 when no fitted detector artifact exists on disk."""
    r_info = client.get("/api/v1/regimes/RELIANCE.NS")
    assert r_info.status_code == 404
    assert r_info.get_json()["error"]["code"] == "REGIME_MODEL_NOT_FOUND"

    r_summary = client.get("/api/v1/regimes/RELIANCE.NS/summary")
    assert r_summary.status_code == 404
    assert r_summary.get_json()["error"]["code"] == "REGIME_MODEL_NOT_FOUND"

    r_latest = client.get("/api/v1/regimes/RELIANCE.NS/latest")
    assert r_latest.status_code == 404
    assert r_latest.get_json()["error"]["code"] == "REGIME_MODEL_NOT_FOUND"


def test_api_v1_regime_endpoints_with_fitted_detector(test_app, client, tmp_path):
    """Test regime endpoints when a fitted detector artifact is available."""
    # Create and fit a minimal synthetic detector
    detector = MarketRegimeDetector(config=RegimeConfig(n_clusters=3, min_train_samples=5))
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "close": np.linspace(100, 120, 15),
        "open": np.linspace(99, 119, 15),
        "high": np.linspace(101, 121, 15),
        "low": np.linspace(98, 118, 15),
        "volume": [1000] * 15,
        "symbol": ["RELIANCE.NS"] * 15,
    })
    detector.fit(df)

    model_file = tmp_path / "regime_detector_reliance_ns.joblib"
    detector.save(str(model_file))

    with patch("backend.app.services.regime_service.RegimeService.find_model_path", return_value=model_file):
        # 1. Info endpoint
        r_info = client.get("/api/v1/regimes/RELIANCE.NS")
        assert r_info.status_code == 200
        info_data = r_info.get_json()["data"]
        assert info_data["n_clusters"] == 3
        assert info_data["model_available"] is True

        # 2. Summary endpoint
        r_sum = client.get("/api/v1/regimes/RELIANCE.NS/summary")
        assert r_sum.status_code == 200
        summaries = r_sum.get_json()["data"]
        assert len(summaries) == 3

        # 3. Latest endpoint
        r_latest = client.get("/api/v1/regimes/RELIANCE.NS/latest")
        assert r_latest.status_code == 200
        latest_data = r_latest.get_json()["data"]
        assert "regime_cluster" in latest_data
        assert "regime_label" in latest_data
        assert "predict future returns" in latest_data["disclaimer"].lower()


# ==============================================================================
# 8. Database Integrity & Cascading Constraints Tests
# ==============================================================================

def test_database_foreign_key_cascading_deletion(test_app):
    """Verify that deleting a BacktestRun cascades to associated trades and snapshots."""
    with test_app.app_context():
        # Create run
        run = BacktestRun(
            name="Cascade Test",
            symbol="RELIANCE.NS",
            strategy_type="buy_and_hold",
            start_date="2024-01-01",
            end_date="2024-01-05",
            initial_capital=100000.0,
            status="completed",
        )
        db.session.add(run)
        db.session.commit()
        run_id = run.id

        # Add trade and snapshot
        trade = TradeRecord(
            backtest_id=run_id,
            symbol="RELIANCE.NS",
            timestamp="2024-01-02T09:15:00Z",
            action="BUY",
            quantity=10.0,
            price=1200.0,
            transaction_cost=10.0,
        )
        snapshot = PortfolioSnapshotRecord(
            backtest_id=run_id,
            timestamp="2024-01-02T15:30:00Z",
            cash=88000.0,
            holdings_value=12000.0,
            portfolio_value=100000.0,
            drawdown=0.0,
        )
        db.session.add_all([trade, snapshot])
        db.session.commit()

        assert TradeRecord.query.filter_by(backtest_id=run_id).count() == 1
        assert PortfolioSnapshotRecord.query.filter_by(backtest_id=run_id).count() == 1

        # Delete backtest run
        db.session.delete(run)
        db.session.commit()

        # Verify child records were cascade deleted
        assert TradeRecord.query.filter_by(backtest_id=run_id).count() == 0
        assert PortfolioSnapshotRecord.query.filter_by(backtest_id=run_id).count() == 0


def test_database_asset_unique_constraint(test_app):
    """Verify unique constraint prevents duplicate asset records."""
    with test_app.app_context():
        # Attempt to insert an already seeded symbol
        dup = Asset(
            symbol="RELIANCE.NS",
            name="Duplicate Reliance",
            exchange="NSE",
            asset_type="EQUITY",
        )
        db.session.add(dup)
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()
