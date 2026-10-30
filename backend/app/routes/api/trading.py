"""Paper Trading & End-to-End Simulation REST API Routes (Phase 9.5B).

Provides endpoints for triggering simulation inference steps, retrieving current
paper trading portfolio balances, positions, executed orders, and risk interventions.
"""
from __future__ import annotations

from flask import request

from ...data.validator import validate_symbol
from ...models.paper_trading import PaperPortfolio, PaperTrade, RiskEventRecord
from ...services.trading_pipeline_service import TradingPipelineService
from . import api_bp, api_error, api_response


@api_bp.route("/trading/step", methods=["POST"])
def execute_trading_step():
    """Execute a single paper trading simulation step using active PPO model & risk validation.
    
    Payload:
        symbol: str (e.g. 'RELIANCE.NS')
        portfolio: Optional[str] (defaults to 'default')
        bar_idx: Optional[int] (specific historical bar index for replay, defaults to latest)
    """
    payload = request.get_json(silent=True) or {}
    symbol = payload.get("symbol")
    if not symbol:
        return api_error(message="Field 'symbol' is required.", code="INVALID_PARAMETER", status_code=400)

    portfolio_name = payload.get("portfolio", "default")
    bar_idx = payload.get("bar_idx")

    service = TradingPipelineService()
    try:
        result = service.execute_simulation_step(
            symbol=symbol,
            portfolio_name=portfolio_name,
            step_bar_idx=bar_idx,
        )
        return api_response(data=result, status_code=200)
    except FileNotFoundError as exc:
        return api_error(message=str(exc), code="MODEL_UNAVAILABLE", status_code=503)
    except ValueError as exc:
        msg = str(exc)
        if "No active or approved PPO model" in msg:
            return api_error(message=msg, code="MODEL_UNAVAILABLE", status_code=503)
        return api_error(message=msg, code="BAD_REQUEST", status_code=400)
    except Exception as exc:
        return api_error(message=f"Simulation step failed: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/trading/portfolios", methods=["GET"])
def list_portfolios():
    """List all available simulation portfolios with summary performance metrics."""
    service = TradingPipelineService()
    portfolios = service.list_portfolios()
    return api_response(data=[p.to_dict(include_positions=True, include_bucket=True) for p in portfolios])


@api_bp.route("/trading/portfolios", methods=["POST"])
def create_portfolio():
    """Create a new paper trading portfolio with configured stock bucket."""
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    if not name or not str(name).strip():
        return api_error(message="Field 'name' is required.", code="INVALID_PARAMETER", status_code=400)

    bucket = payload.get("bucket", [])
    if not bucket:
        return api_error(message="Stock bucket must contain at least one stock.", code="EMPTY_BUCKET", status_code=400)

    initial_cash = payload.get("initial_cash", 100000.0)
    alloc_method = payload.get("allocation_method", "EQUAL_WEIGHT")
    strategy_type = payload.get("strategy_type", "BUY_AND_HOLD")
    risk_config = payload.get("risk_config", {})
    description = payload.get("description")
    user_id = payload.get("user_id", "default_user")

    service = TradingPipelineService()
    try:
        portfolio = service.create_portfolio(
            name=name,
            initial_cash=initial_cash,
            bucket=bucket,
            allocation_method=alloc_method,
            strategy_type=strategy_type,
            risk_config=risk_config,
            description=description,
            user_id=user_id,
        )
        return api_response(data=portfolio.to_dict(include_positions=True, include_bucket=True), status_code=201)
    except ValueError as exc:
        return api_error(message=str(exc), code="VALIDATION_ERROR", status_code=400)
    except Exception as exc:
        return api_error(message=f"Failed to create portfolio: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/trading/portfolios/<name_or_id>", methods=["GET"])
def get_portfolio_detail(name_or_id: str):
    """Retrieve detailed state of a specific portfolio."""
    service = TradingPipelineService()
    portfolio = service.get_portfolio(name_or_id)
    if not portfolio:
        return api_error(message=f"Portfolio '{name_or_id}' not found.", code="NOT_FOUND", status_code=404)
    return api_response(data=portfolio.to_dict(include_positions=True, include_bucket=True))


@api_bp.route("/trading/portfolios/<name_or_id>/bucket", methods=["PUT"])
def update_portfolio_bucket(name_or_id: str):
    """Update stock bucket selection and target allocations for a portfolio."""
    payload = request.get_json(silent=True) or {}
    bucket = payload.get("bucket")
    if not bucket:
        return api_error(message="Field 'bucket' is required and cannot be empty.", code="INVALID_PARAMETER", status_code=400)

    alloc_method = payload.get("allocation_method")
    service = TradingPipelineService()
    try:
        portfolio = service.update_portfolio_bucket(
            name_or_id=name_or_id,
            bucket=bucket,
            allocation_method=alloc_method,
        )
        return api_response(data=portfolio.to_dict(include_positions=True, include_bucket=True))
    except ValueError as exc:
        return api_error(message=str(exc), code="VALIDATION_ERROR", status_code=400)
    except Exception as exc:
        return api_error(message=f"Failed to update bucket: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/trading/portfolios/<name_or_id>/rebalance", methods=["POST"])
def rebalance_portfolio(name_or_id: str):
    """Execute simulated rebalancing across configured stock bucket assets."""
    payload = request.get_json(silent=True) or {}
    bar_idx = payload.get("bar_idx")
    service = TradingPipelineService()
    try:
        result = service.rebalance_portfolio(name_or_id=name_or_id, step_bar_idx=bar_idx)
        return api_response(data=result, status_code=200)
    except ValueError as exc:
        return api_error(message=str(exc), code="BAD_REQUEST", status_code=400)
    except Exception as exc:
        return api_error(message=f"Rebalance failed: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/trading/portfolios/<name_or_id>/snapshots", methods=["GET"])
def get_portfolio_snapshots(name_or_id: str):
    """Retrieve historical valuation snapshots for plotting the portfolio equity curve."""
    service = TradingPipelineService()
    portfolio = service.get_portfolio(name_or_id)
    if not portfolio:
        return api_error(message=f"Portfolio '{name_or_id}' not found.", code="NOT_FOUND", status_code=404)

    snapshots = service.get_portfolio_snapshots(portfolio.id)
    return api_response(data=[s.to_dict() for s in snapshots])


@api_bp.route("/trading/portfolios/<name_or_id>", methods=["DELETE"])
def delete_portfolio(name_or_id: str):
    """Delete a custom portfolio."""
    service = TradingPipelineService()
    try:
        service.delete_portfolio(name_or_id)
        return api_response(data={"deleted": True, "name_or_id": name_or_id})
    except ValueError as exc:
        return api_error(message=str(exc), code="BAD_REQUEST", status_code=400)
    except Exception as exc:
        return api_error(message=f"Failed to delete portfolio: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/trading/portfolio", methods=["GET"])
def get_paper_portfolio():
    """Retrieve current paper trading portfolio valuation, cash, and open positions."""
    name = request.args.get("name") or request.args.get("portfolio") or "default"
    service = TradingPipelineService()
    portfolio = service.get_portfolio(name)
    if not portfolio:
        portfolio = service.get_or_create_portfolio(name=name)

    return api_response(data=portfolio.to_dict(include_positions=True, include_bucket=True))


@api_bp.route("/trading/trades", methods=["GET"])
def list_paper_trades():
    """List paginated executed simulation trades."""
    name = request.args.get("name") or request.args.get("portfolio") or "default"
    service = TradingPipelineService()
    portfolio = service.get_portfolio(name)
    if not portfolio:
        return api_response(data=[], meta={"count": 0, "page": 1, "total_pages": 0})

    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        return api_error(message="page and limit must be positive integers.", status_code=400)

    query = PaperTrade.query.filter_by(portfolio_id=portfolio.id).order_by(PaperTrade.id.desc())
    total_count = query.count()
    total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
    trades = query.offset((page - 1) * limit).limit(limit).all()

    return api_response(
        data=[t.to_dict() for t in trades],
        meta={"count": len(trades), "total_count": total_count, "page": page, "total_pages": total_pages},
    )


@api_bp.route("/trading/risk-events", methods=["GET"])
def list_risk_events():
    """List paginated pre-trade risk management intervention events."""
    symbol = request.args.get("symbol")
    portfolio_param = request.args.get("portfolio") or request.args.get("name")

    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        return api_error(message="page and limit must be positive integers.", status_code=400)

    query = RiskEventRecord.query
    if portfolio_param:
        service = TradingPipelineService()
        portfolio = service.get_portfolio(portfolio_param)
        if portfolio:
            query = query.filter_by(portfolio_id=portfolio.id)

    if symbol:
        try:
            clean_sym = validate_symbol(symbol)
            query = query.filter_by(symbol=clean_sym)
        except Exception:
            pass

    query = query.order_by(RiskEventRecord.id.desc())
    total_count = query.count()
    total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
    events = query.offset((page - 1) * limit).limit(limit).all()

    return api_response(
        data=[e.to_dict() for e in events],
        meta={"count": len(events), "total_count": total_count, "page": page, "total_pages": total_pages},
    )


# =============================================================================
# Phase 13: Automated Paper Trading Engine Endpoints
# =============================================================================

def get_paper_service():
    """Helper to retrieve PaperTradingService singleton."""
    from ...services.paper_trading_service import PaperTradingService
    return PaperTradingService.get_instance()


@api_bp.route("/trading/portfolios/<name_or_id>/start", methods=["POST"])
def start_portfolio_paper_trading(name_or_id: str):
    """Enable automated paper trading for a portfolio."""
    service = get_paper_service()
    success = service.enable_paper_trading(name_or_id)
    if not success:
        return api_error(message=f"Portfolio '{name_or_id}' not found.", code="NOT_FOUND", status_code=404)
    return api_response(
        data={"portfolio": name_or_id, "is_active": True},
        meta={"message": f"Automated paper trading started for portfolio '{name_or_id}'."},
    )


@api_bp.route("/trading/portfolios/<name_or_id>/stop", methods=["POST"])
def stop_portfolio_paper_trading(name_or_id: str):
    """Pause automated paper trading for a portfolio."""
    service = get_paper_service()
    success = service.disable_paper_trading(name_or_id)
    if not success:
        return api_error(message=f"Portfolio '{name_or_id}' not found.", code="NOT_FOUND", status_code=404)
    return api_response(
        data={"portfolio": name_or_id, "is_active": False},
        meta={"message": f"Automated paper trading paused for portfolio '{name_or_id}'."},
    )


@api_bp.route("/trading/portfolios/<name_or_id>/status", methods=["GET"])
def get_portfolio_paper_status(name_or_id: str):
    """Retrieve paper-trading status, active state, cash, exposure, and performance."""
    service = get_paper_service()
    status = service.get_portfolio_status(name_or_id)
    if not status:
        return api_error(message=f"Portfolio '{name_or_id}' not found.", code="NOT_FOUND", status_code=404)
    return api_response(data=status)


@api_bp.route("/trading/portfolios/<name_or_id>/positions", methods=["GET"])
def get_portfolio_positions(name_or_id: str):
    """Retrieve active open positions and market values for a portfolio."""
    service = get_paper_service()
    positions = service.get_positions(name_or_id)
    return api_response(data=positions, meta={"count": len(positions)})


@api_bp.route("/trading/portfolios/<name_or_id>/trades", methods=["GET"])
def get_portfolio_trades(name_or_id: str):
    """Retrieve paginated trade execution history for a portfolio."""
    service = get_paper_service()
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        return api_error(message="page and limit must be positive integers.", status_code=400)

    trades_data = service.get_trades(name_or_id, page=page, limit=limit)
    return api_response(
        data=trades_data["trades"],
        meta={
            "count": len(trades_data["trades"]),
            "total_count": trades_data["total"],
            "page": trades_data["page"],
            "total_pages": trades_data["total_pages"],
        },
    )


@api_bp.route("/trading/signals/<signal_id>/trade", methods=["GET"])
def get_signal_trade_lineage(signal_id: str):
    """Retrieve executed simulated trade associated with a specific PPO signal_id."""
    service = get_paper_service()
    trade = service.get_trade_by_signal_id(signal_id)
    if not trade:
        return api_error(
            message=f"No executed paper trade found for signal '{signal_id}'.",
            code="NOT_FOUND",
            status_code=404,
        )
    return api_response(data=trade)


@api_bp.route("/trading/execute-signal", methods=["POST"])
def execute_signal_manually():
    """Explicitly execute a validated signal against a portfolio."""
    payload = request.get_json(silent=True) or {}
    signal_ref = payload.get("signal_id") or payload.get("signal")
    if not signal_ref:
        return api_error(message="Field 'signal_id' or 'signal' is required.", code="MISSING_PARAM", status_code=400)

    portfolio_ref = payload.get("portfolio") or payload.get("portfolio_id")
    service = get_paper_service()

    result = service.process_signal(signal=signal_ref, portfolio_name_or_id=portfolio_ref)
    return api_response(data=result)


@api_bp.route("/trading/engine/health", methods=["GET"])
def get_trading_engine_health():
    """Retrieve paper-trading execution engine health, transaction costs, and telemetry."""
    service = get_paper_service()
    health = service.get_engine_health()
    return api_response(data=health)
