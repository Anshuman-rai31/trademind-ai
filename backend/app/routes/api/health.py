"""Health and System Status REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

from flask import current_app
from sqlalchemy import text

from ...config import PROJECT_ROOT
from ...db import db
from ...services.agent_service import AgentService
from ...services.market_data_service import MarketDataService
from ...services.regime_service import RegimeService
from ...services.risk_service import RiskService
from . import api_bp, api_error, api_response


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint confirming API status and database connectivity."""
    db_connected = False
    try:
        db.session.execute(text("SELECT 1"))
        db_connected = True
    except Exception as exc:
        current_app.logger.warning("Database connectivity check failed: %s", exc)
        db_connected = False

    data = {
        "status": "healthy" if db_connected else "degraded",
        "app_name": current_app.config.get("APP_NAME", "TradeMind"),
        "version": current_app.config.get("APP_VERSION", "0.9.0"),
        "phase": current_app.config.get("PHASE", "Phase 9: Flask API & Database Integration"),
        "environment": current_app.config.get("ENV", "development"),
        "database": {
            "connected": db_connected,
            "engine": "sqlite",
        },
    }

    if not db_connected:
        return api_error(
            message="Database service is unavailable.",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=data,
        )

    return api_response(data=data, status_code=200)


@api_bp.route("/system/status", methods=["GET"])
def system_status():
    """Concise capability summary across data, baseline, ML, risk, and regime modules."""
    data_service = MarketDataService()
    agent_service = AgentService()
    risk_service = RiskService()
    regime_service = RegimeService()

    # 1. Market Data Capabilities
    processed_files = list(data_service.processed_dir.glob("*.csv"))
    market_data_info = {
        "market": "NSE",
        "data_source": "yfinance",
        "frequency": "1d",
        "processed_datasets_count": len(processed_files),
        "supported_assets_count": len(data_service.list_assets()),
    }

    # 2. Baseline Strategies
    baseline_strategies = [
        {
            "id": "buy_and_hold",
            "name": "Buy and Hold",
            "description": "Baseline benchmark executing a single long position at the start of evaluation.",
        },
        {
            "id": "ma_crossover",
            "name": "Moving Average Crossover",
            "description": "Trend-following strategy generating signals on fast and slow moving average crosses.",
        },
    ]

    # 3. PPO Agent Capabilities & Artifact Readiness
    agent_status = agent_service.get_agent_status()

    # 4. Risk Management Capability
    risk_status = risk_service.get_status()

    # 5. Regime Detection Capability
    saved_regime_models = list((PROJECT_ROOT / "models").glob("regime_detector*.joblib"))
    regime_status = {
        "installed": True,
        "algorithm": "K-Means Clustering",
        "artifacts_available": len(saved_regime_models) > 0,
        "saved_models_count": len(saved_regime_models),
    }

    # 6. Model Registry & Paper Trading Capabilities
    from ...models.model_registry import RegisteredModel
    from ...models.paper_trading import PaperPortfolio

    model_registry_info = {
        "installed": True,
        "registered_models_count": RegisteredModel.query.count(),
        "active_models_count": RegisteredModel.query.filter_by(is_active=True).count(),
    }
    simulation_trading_info = {
        "installed": True,
        "mode": "Simulation (Paper Trading)",
        "active_portfolios_count": PaperPortfolio.query.count(),
    }

    status_data = {
        "market_data": market_data_info,
        "baseline_strategies": baseline_strategies,
        "ppo_agent": agent_status,
        "risk_management": risk_status,
        "market_regime": regime_status,
        "model_registry": model_registry_info,
        "simulation_trading": simulation_trading_info,
    }

    return api_response(data=status_data, status_code=200)
