"""Dashboard Summary REST API Route for TradeMind (Phase 9)."""

from __future__ import annotations

from ...config import PROJECT_ROOT
from ...models.asset import Asset
from ...models.backtest import BacktestRun, PortfolioSnapshotRecord
from ...services.agent_service import AgentService
from ...services.risk_service import RiskService
from . import api_bp, api_response


@api_bp.route("/dashboard/summary", methods=["GET"])
def dashboard_summary():
    """Return platform dashboard summary assembled from actual persisted data."""
    # 1. Assets count
    assets_count = Asset.query.count()

    # 2. Backtests summary
    completed_backtests_count = BacktestRun.query.filter_by(status="completed").count()
    latest_backtest_record = (
        BacktestRun.query.filter_by(status="completed")
        .order_by(BacktestRun.completed_at.desc(), BacktestRun.id.desc())
        .first()
    )

    latest_backtest_data = None
    latest_valuation = None

    if latest_backtest_record:
        latest_backtest_data = latest_backtest_record.to_dict(include_metrics=True)

        # Get latest snapshot for portfolio valuation
        latest_snapshot = (
            PortfolioSnapshotRecord.query.filter_by(backtest_id=latest_backtest_record.id)
            .order_by(PortfolioSnapshotRecord.id.desc())
            .first()
        )
        if latest_snapshot:
            latest_valuation = {
                "backtest_id": latest_backtest_record.id,
                "strategy_name": latest_backtest_record.name,
                "timestamp": latest_snapshot.timestamp,
                "portfolio_nav": latest_snapshot.portfolio_value,
                "cash": latest_snapshot.cash,
                "holdings_value": latest_snapshot.holdings_value,
                "drawdown": latest_snapshot.drawdown,
            }

    # 3. PPO Agent readiness and metrics
    agent_service = AgentService()
    agent_status = agent_service.get_agent_status()
    latest_agent_metrics = agent_service.get_persisted_metrics()

    # 4. Risk configuration
    risk_service = RiskService()
    risk_config = risk_service.get_config()

    # 5. Market regime information
    saved_regime_models = list((PROJECT_ROOT / "models").glob("regime_detector*.joblib"))
    regime_info = {
        "available": len(saved_regime_models) > 0,
        "models_count": len(saved_regime_models),
    }

    summary_data = {
        "supported_assets_count": assets_count,
        "completed_backtests_count": completed_backtests_count,
        "latest_completed_backtest": latest_backtest_data,
        "latest_portfolio_valuation": latest_valuation,
        "ppo_agent": {
            "training_installed": agent_status["training_support_installed"],
            "artifact_available": agent_status["saved_model_artifact_available"],
            "ready_for_inference": agent_status["ready_for_inference"],
            "model_path": agent_status["model_artifact_path"],
            "metrics": latest_agent_metrics,
        },
        "risk_configuration": risk_config,
        "market_regime": regime_info,
    }

    return api_response(data=summary_data, status_code=200)
