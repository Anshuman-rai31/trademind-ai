"""Backtesting REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

import re
from typing import Optional

from flask import current_app, request

from ...data.validator import validate_symbol
from ...db import db, safe_json_loads
from ...models.backtest import BacktestRun, PortfolioSnapshotRecord, TradeRecord
from ...services.backtest_service import BacktestService
from . import api_bp, api_error, api_response


@api_bp.route("/backtests", methods=["POST"])
def create_backtest():
    """Create and queue a backtest simulation run."""
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return api_error(
            message="Request body must be a valid JSON object.",
            code="MALFORMED_JSON",
            status_code=400,
        )

    name = payload.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        return api_error(
            message="Field 'name' is required and must be a non-empty string.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    symbol = payload.get("symbol")
    if not symbol or not isinstance(symbol, str):
        return api_error(
            message="Field 'symbol' is required.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    try:
        clean_sym = validate_symbol(symbol)
    except Exception as exc:
        return api_error(
            message=f"Invalid symbol '{symbol}': {exc}",
            code="INVALID_SYMBOL",
            status_code=400,
        )

    strategy_type = payload.get("strategy_type")
    if not strategy_type or not isinstance(strategy_type, str):
        return api_error(
            message="Field 'strategy_type' is required (e.g. 'buy_and_hold', 'ma_crossover').",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    strategy_clean = strategy_type.strip().lower()
    allowed_strategies = {"buy_and_hold", "ma_crossover", "ppo"}
    if strategy_clean not in allowed_strategies:
        return api_error(
            message=f"Unsupported strategy '{strategy_type}'. Allowed: {sorted(allowed_strategies)}",
            code="UNSUPPORTED_STRATEGY",
            status_code=400,
        )

    # Initial Capital
    raw_capital = payload.get("initial_capital", 100000.0)
    try:
        initial_capital = float(raw_capital)
        if initial_capital <= 0:
            return api_error(
                message="Field 'initial_capital' must be strictly positive.",
                code="INVALID_PARAMETER",
                status_code=400,
            )
    except (ValueError, TypeError):
        return api_error(
            message="Field 'initial_capital' must be a valid number.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    # Date format validation
    date_regex = r"^\d{4}-\d{2}-\d{2}$"
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")

    if start_date:
        if not isinstance(start_date, str) or not re.match(date_regex, start_date.strip()):
            return api_error(
                message="start_date must be in YYYY-MM-DD format.",
                code="INVALID_DATE_FORMAT",
                status_code=400,
            )
        start_date = start_date.strip()

    if end_date:
        if not isinstance(end_date, str) or not re.match(date_regex, end_date.strip()):
            return api_error(
                message="end_date must be in YYYY-MM-DD format.",
                code="INVALID_DATE_FORMAT",
                status_code=400,
            )
        end_date = end_date.strip()

    if start_date and end_date and start_date >= end_date:
        return api_error(
            message=f"start_date ({start_date}) must be earlier than end_date ({end_date}).",
            code="INVALID_DATE_RANGE",
            status_code=400,
        )

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        return api_error(
            message="Optional field 'parameters' must be a dictionary.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    sync_flag = request.args.get("sync", "false").lower() in ("true", "1", "yes")

    backtest_service = BacktestService()
    try:
        run = backtest_service.create_and_queue_backtest(
            app=current_app._get_current_object(),
            name=name.strip(),
            symbol=clean_sym,
            strategy_type=strategy_clean,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            parameters=parameters,
            sync=sync_flag,
        )
    except ValueError as exc:
        return api_error(message=str(exc), code="INVALID_BACKTEST_CONFIG", status_code=400)
    except Exception as exc:
        return api_error(
            message=f"Failed to initialize backtest: {exc}",
            code="SERVER_ERROR",
            status_code=500,
        )

    status_code = 201 if run.status == "completed" else 202
    return api_response(data=run.to_dict(include_metrics=True), status_code=status_code)


@api_bp.route("/backtests", methods=["GET"])
def list_backtests():
    """List persisted backtests with filtering and pagination."""
    symbol_filter = request.args.get("symbol")
    status_filter = request.args.get("status")
    page_str = request.args.get("page", "1")
    limit_str = request.args.get("limit", "20")

    try:
        page = max(1, int(page_str))
        limit = min(100, max(1, int(limit_str)))
    except ValueError:
        return api_error(
            message="Parameters 'page' and 'limit' must be positive integers.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    query = BacktestRun.query

    if symbol_filter:
        try:
            clean_sym = validate_symbol(symbol_filter)
            query = query.filter_by(symbol=clean_sym)
        except Exception:
            query = query.filter_by(symbol=symbol_filter.strip().upper())

    if status_filter:
        query = query.filter_by(status=status_filter.strip().lower())

    total = query.count()
    runs = query.order_by(BacktestRun.id.desc()).offset((page - 1) * limit).limit(limit).all()

    items = [r.to_dict(include_metrics=False) for r in runs]
    meta = {
        "page": page,
        "limit": limit,
        "total": total,
        "pages": (total + limit - 1) // limit if total > 0 else 1,
    }
    return api_response(data=items, meta=meta, status_code=200)


@api_bp.route("/backtests/<int:backtest_id>", methods=["GET"])
def get_backtest_details(backtest_id: int):
    """Return detailed metadata, execution parameters, and metrics for a backtest."""
    run: Optional[BacktestRun] = db.session.get(BacktestRun, backtest_id)
    if not run:
        return api_error(
            message=f"Backtest with ID {backtest_id} not found.",
            code="RESOURCE_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=run.to_dict(include_metrics=True), status_code=200)


@api_bp.route("/backtests/<int:backtest_id>/trades", methods=["GET"])
def get_backtest_trades(backtest_id: int):
    """Return persisted trade executions with pagination."""
    run: Optional[BacktestRun] = db.session.get(BacktestRun, backtest_id)
    if not run:
        return api_error(
            message=f"Backtest with ID {backtest_id} not found.",
            code="RESOURCE_NOT_FOUND",
            status_code=404,
        )

    page_str = request.args.get("page", "1")
    limit_str = request.args.get("limit", "50")
    try:
        page = max(1, int(page_str))
        limit = min(500, max(1, int(limit_str)))
    except ValueError:
        return api_error(
            message="Parameters 'page' and 'limit' must be positive integers.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    query = TradeRecord.query.filter_by(backtest_id=backtest_id)
    total = query.count()
    trades = query.order_by(TradeRecord.id.asc()).offset((page - 1) * limit).limit(limit).all()

    items = [t.to_dict() for t in trades]
    meta = {
        "backtest_id": backtest_id,
        "page": page,
        "limit": limit,
        "total": total,
        "pages": (total + limit - 1) // limit if total > 0 else 1,
    }
    return api_response(data=items, meta=meta, status_code=200)


@api_bp.route("/backtests/<int:backtest_id>/equity-curve", methods=["GET"])
def get_backtest_equity_curve(backtest_id: int):
    """Return portfolio mark-to-market valuation series generated by simulator."""
    run: Optional[BacktestRun] = db.session.get(BacktestRun, backtest_id)
    if not run:
        return api_error(
            message=f"Backtest with ID {backtest_id} not found.",
            code="RESOURCE_NOT_FOUND",
            status_code=404,
        )

    snapshots = (
        PortfolioSnapshotRecord.query.filter_by(backtest_id=backtest_id)
        .order_by(PortfolioSnapshotRecord.id.asc())
        .all()
    )

    items = [s.to_dict() for s in snapshots]
    meta = {
        "backtest_id": backtest_id,
        "symbol": run.symbol,
        "strategy_type": run.strategy_type,
        "total_snapshots": len(items),
    }
    return api_response(data=items, meta=meta, status_code=200)


@api_bp.route("/backtests/<int:backtest_id>/metrics", methods=["GET"])
def get_backtest_metrics(backtest_id: int):
    """Return actual computed institutional performance metrics."""
    run: Optional[BacktestRun] = db.session.get(BacktestRun, backtest_id)
    if not run:
        return api_error(
            message=f"Backtest with ID {backtest_id} not found.",
            code="RESOURCE_NOT_FOUND",
            status_code=404,
        )

    if run.status != "completed":
        return api_response(
            data={
                "backtest_id": run.id,
                "status": run.status,
                "error_message": run.error_message,
                "metrics": None,
            },
            meta={"message": f"Backtest status is '{run.status}'. Metrics available once completed."},
            status_code=200,
        )

    metrics = safe_json_loads(run.metrics_json, {})
    meta = {
        "backtest_id": run.id,
        "symbol": run.symbol,
        "strategy_type": run.strategy_type,
    }
    return api_response(data=metrics, meta=meta, status_code=200)
