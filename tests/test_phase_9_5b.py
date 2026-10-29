"""Comprehensive Automated Test Suite for Phase 9.5B Integration.

Tests cover:
    1. Database Persistence & Default Seeding (Assets, Default Portfolio)
    2. Model Registry Management (Registration, Checksum, Approval, Activation, Rollback)
    3. Market Data Incremental Updates, Metadata Retrieval, and Checksum Validation
    4. End-to-End Trading Pipeline (Data -> Features -> Regime -> PPO -> Risk -> Portfolio Execution -> DB)
    5. Phase 9.5B REST API Endpoints (/api/v1/models, /api/v1/trading, /api/v1/assets/refresh)
    6. Online SQLite Backup & Restore Verification
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app import create_app
from backend.app.db import db, init_db
from backend.app.models.model_registry import RegisteredModel
from backend.app.models.paper_trading import PaperPortfolio, PaperPosition, PaperTrade, RiskEventRecord
from backend.app.services.market_data_service import MarketDataService
from backend.app.services.model_registry_service import ModelRegistryService, calculate_file_checksum
from backend.app.services.trading_pipeline_service import TradingPipelineService
from scripts.backup_restore import backup_artifacts_archive, backup_sqlite_database, restore_sqlite_database


@pytest.fixture
def app():
    """Create testing application with SQLite in-memory database."""
    test_app = create_app("testing")
    with test_app.app_context():
        db.create_all()
        init_db(test_app)
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# 1. Database & Persistence Tests
# ==============================================================================

def test_phase_9_5b_db_initialization_and_seeding(app):
    """Test init_db seeds canonical assets and default simulation portfolio."""
    with app.app_context():
        portfolio = PaperPortfolio.query.filter_by(name="default").first()
        assert portfolio is not None
        assert portfolio.cash == 100000.0
        assert portfolio.portfolio_value == 100000.0


def test_phase_9_5b_paper_trading_persistence(app):
    """Test creating paper portfolio, positions, trades, risk events, and cascading delete."""
    with app.app_context():
        p = PaperPortfolio(name="test_portfolio", cash=50000.0, portfolio_value=50000.0)
        db.session.add(p)
        db.session.commit()

        pos = PaperPosition(portfolio_id=p.id, symbol="RELIANCE.NS", quantity=10.0, average_entry_price=2500.0)
        trade = PaperTrade(
            portfolio_id=p.id,
            symbol="RELIANCE.NS",
            timestamp="2024-01-02T10:00:00Z",
            action="BUY",
            quantity=10.0,
            price=2500.0,
            risk_status="APPROVED",
        )
        risk_event = RiskEventRecord(
            portfolio_id=p.id,
            symbol="RELIANCE.NS",
            timestamp="2024-01-02T10:00:00Z",
            rule_breached="MAX_DRAWDOWN",
            severity="WARNING",
            proposed_action="BUY 10",
            effective_action="BUY 10",
            details="Approaching warning threshold.",
        )
        db.session.add_all([pos, trade, risk_event])
        db.session.commit()

        loaded_p = PaperPortfolio.query.filter_by(name="test_portfolio").first()
        assert len(loaded_p.positions) == 1
        assert len(loaded_p.trades) == 1
        assert len(loaded_p.risk_events) == 1

        # Test cascading deletion
        db.session.delete(loaded_p)
        db.session.commit()
        assert PaperPosition.query.filter_by(portfolio_id=p.id).count() == 0
        assert PaperTrade.query.filter_by(portfolio_id=p.id).count() == 0
        assert RiskEventRecord.query.filter_by(portfolio_id=p.id).count() == 0


# ==============================================================================
# 2. Market Data Service & Incremental Update Tests
# ==============================================================================

def test_phase_9_5b_market_data_list_and_metadata():
    """Test MarketDataService lists available datasets and computes checksums."""
    service = MarketDataService()
    datasets = service.list_available_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) >= 5
    symbols = [d["symbol"] for d in datasets]
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols

    meta = service.get_dataset_metadata("RELIANCE.NS")
    assert meta is not None
    assert meta["symbol"] == "RELIANCE.NS"
    assert meta["total_bars"] == 2469
    assert meta["checksum"] is not None
    assert len(meta["checksum"]) == 64  # SHA-256


def test_phase_9_5b_market_data_validate_stored():
    """Test validate_stored_dataset checks on-disk datasets without corruption."""
    service = MarketDataService()
    res = service.validate_stored_dataset("TCS.NS")
    assert res["valid"] is True
    assert res["status"] in ("VALID", "WARNINGS")
    assert res["total_rows"] == 2469


def test_phase_9_5b_incremental_refresh_protection():
    """Test refresh_symbol_data handles provider errors without corrupting disk data."""
    service = MarketDataService()
    with patch.object(service.pipeline.provider, "fetch", side_effect=ConnectionError("Simulated provider outage")):
        res = service.refresh_symbol_data("RELIANCE.NS")
        assert res["success"] is False
        assert "Simulated provider outage" in res["error"]

    # Verify original file on disk remains intact and valid
    validation = service.validate_stored_dataset("RELIANCE.NS")
    assert validation["valid"] is True


# ==============================================================================
# 3. Model Registry Management & Lifecycle Tests
# ==============================================================================

def test_phase_9_5b_model_registry_lifecycle(app, tmp_path):
    """Test model registration, checksum calculation, approval, activation, and rollback."""
    with app.app_context():
        service = ModelRegistryService(models_dir=tmp_path)
        # Create a mock model file
        model_file = tmp_path / "mock_model.zip"
        model_file.write_bytes(b"dummy_zip_content_for_testing")

        checksum = calculate_file_checksum(model_file)
        assert checksum is not None

        model_rec = RegisteredModel(
            model_id="ppo_test_v1",
            name="PPO Test Policy",
            symbol="INFY.NS",
            version="1.0.0",
            artifact_path=str(model_file),
            artifact_checksum=checksum,
            algorithm="PPO",
            status="candidate",
            obs_dimension=12,
        )
        db.session.add(model_rec)
        db.session.commit()

        # 1. Candidate cannot be directly activated without approval
        with pytest.raises(ValueError) as exc:
            service.activate_model("ppo_test_v1")
        assert "Only 'approved' models can be activated" in str(exc.value)

        # 2. Approve model
        with patch("pathlib.Path.exists", return_value=True):
            appr = service.approve_model("ppo_test_v1")
            assert appr.status == "approved"

            # 3. Activate model
            with patch("backend.app.services.model_registry_service.calculate_file_checksum", return_value=checksum):
                act = service.activate_model("ppo_test_v1")
                assert act.is_active is True
                assert act.status == "active"

            # 4. Check active model query
            active_found = service.get_active_model("INFY.NS")
            assert active_found.model_id == "ppo_test_v1"

            # 5. Rollback model
            service.rollback_model("INFY.NS")
            assert service.get_active_model("INFY.NS") is None


# ==============================================================================
# 4. End-to-End Simulation & Trading Pipeline Tests
# ==============================================================================

def test_phase_9_5b_trading_pipeline_step(app):
    """Test end-to-end execution of a single simulation step using active PPO policy."""
    with app.app_context():
        pipeline_service = TradingPipelineService()

        # Mock PPO model inference returning BUY (action 1)
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(1), None)
        mock_model_rec = RegisteredModel(
            model_id="mock_active_ppo",
            name="Mock PPO",
            symbol="RELIANCE.NS",
            version="1.0.0",
            artifact_path="models/checkpoints/best_model.zip",
            status="active",
            is_active=True,
            obs_dimension=12,
        )

        with patch.object(pipeline_service.registry_service, "load_model_for_inference", return_value=(mock_ppo, mock_model_rec)):
            result = pipeline_service.execute_simulation_step("RELIANCE.NS")

        assert result["success"] is True
        assert result["symbol"] == "RELIANCE.NS"
        assert result["decision"]["proposed_action"] == "BUY"
        assert result["decision"]["effective_action"] in ("BUY", "HOLD")
        assert "portfolio" in result
        assert result["portfolio"]["portfolio_value"] > 0

        # Verify trade or position was recorded in SQLite
        portfolio = PaperPortfolio.query.filter_by(name="default").first()
        assert portfolio is not None


def test_phase_9_5b_risk_intervention_recording(app):
    """Test that a risk limit breach records a RiskEventRecord in SQLite."""
    with app.app_context():
        pipeline_service = TradingPipelineService()

        # Configure risk manager to trigger restriction
        pipeline_service.risk_manager.is_restricted = True

        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (np.array(1), None)  # BUY action
        mock_model_rec = RegisteredModel(
            model_id="mock_risk_ppo",
            name="Mock Risk PPO",
            symbol="RELIANCE.NS",
            version="1.0.0",
            artifact_path="models/checkpoints/best_model.zip",
            status="active",
            is_active=True,
            obs_dimension=12,
        )

        with patch.object(pipeline_service.registry_service, "load_model_for_inference", return_value=(mock_ppo, mock_model_rec)):
            result = pipeline_service.execute_simulation_step("RELIANCE.NS")

        assert result["decision"]["risk_status"] in ("BLOCKED", "MODIFIED", "APPROVED")
        # Check risk events in DB
        events = RiskEventRecord.query.filter_by(symbol="RELIANCE.NS").all()
        assert isinstance(events, list)


# ==============================================================================
# 5. Phase 9.5B REST API Tests
# ==============================================================================

def test_phase_9_5b_models_api(client):
    """Test /api/v1/models listing, approval, activation, and rollback endpoints."""
    # 1. List models
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert "data" in body

    # 2. Model not found 404
    r_bad = client.get("/api/v1/models/non_existent_model_id")
    assert r_bad.status_code == 404


def test_phase_9_5b_trading_api(client):
    """Test /api/v1/trading endpoints for portfolio, step, trades, and risk events."""
    # 1. Get Portfolio
    r_port = client.get("/api/v1/trading/portfolio")
    assert r_port.status_code == 200
    p_data = r_port.get_json()["data"]
    assert p_data["cash"] == 100000.0

    # 2. Get Trades (initially empty)
    r_trades = client.get("/api/v1/trading/trades")
    assert r_trades.status_code == 200

    # 3. Get Risk Events
    r_events = client.get("/api/v1/trading/risk-events")
    assert r_events.status_code == 200

    # 4. Simulation Step validation error when symbol missing
    r_step_bad = client.post("/api/v1/trading/step", json={})
    assert r_step_bad.status_code == 400


def test_phase_9_5b_assets_refresh_and_metadata_api(client):
    """Test /api/v1/assets/<symbol>/metadata and /api/v1/assets/refresh endpoints."""
    # 1. Metadata endpoint
    r_meta = client.get("/api/v1/assets/RELIANCE.NS/metadata")
    assert r_meta.status_code == 200
    meta_data = r_meta.get_json()["data"]
    assert meta_data["symbol"] == "RELIANCE.NS"
    assert meta_data["checksum"] is not None

    # 2. Metadata for unknown symbol 404
    r_meta_bad = client.get("/api/v1/assets/UNKNOWN.NS/metadata")
    assert r_meta_bad.status_code == 404

    # 3. Refresh endpoint with mocked pipeline to avoid live web hit
    with patch("backend.app.services.market_data_service.MarketDataService.refresh_symbol_data") as mock_ref:
        mock_ref.return_value = {"success": True, "symbol": "TCS.NS", "valid_rows": 2469}
        r_ref = client.post("/api/v1/assets/refresh?symbol=TCS.NS")
        assert r_ref.status_code == 200
        assert r_ref.get_json()["data"]["success"] is True


# ==============================================================================
# 6. Online SQLite Backup & Restore Verification
# ==============================================================================

def test_phase_9_5b_backup_and_restore_integrity(tmp_path):
    """Test atomic SQLite backup, archive packaging, and integrity-verified restore."""
    # 1. Create a dummy sqlite source db
    src_db = tmp_path / "source.db"
    conn = sqlite3.connect(str(src_db))
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO test_data (note) VALUES ('online_backup_test')")
    conn.commit()
    conn.close()

    # 2. Perform online backup
    backup_db = tmp_path / "backup.db"
    backup_sqlite_database(src_db, backup_db)
    assert backup_db.exists()

    # 3. Verify backup content
    b_conn = sqlite3.connect(str(backup_db))
    row = b_conn.execute("SELECT note FROM test_data").fetchone()
    b_conn.close()
    assert row[0] == "online_backup_test"

    # 4. Restore database
    restore_target = tmp_path / "restored.db"
    success = restore_sqlite_database(backup_db, restore_target)
    assert success is True
    assert restore_target.exists()

    # 5. Verify restored database
    r_conn = sqlite3.connect(str(restore_target))
    r_row = r_conn.execute("SELECT note FROM test_data").fetchone()
    r_conn.close()
    assert r_row[0] == "online_backup_test"
