"""RL Trading Signals REST API Routes (Phase 12).

Provides endpoints for monitoring model-generated BUY, SELL, and HOLD trading signals,
retrieving paginated signal history, inspecting detailed observation vectors and risk interventions,
checking signal engine health and model readiness, and triggering on-demand candle evaluations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from flask import request

from ...data.validator import validate_symbol
from ...signals.service import RealtimeSignalService
from . import api_bp, api_error, api_response


def get_signal_service() -> RealtimeSignalService:
    """Helper to retrieve singleton signal service."""
    return RealtimeSignalService.get_instance()


@api_bp.route("/signals/latest", methods=["GET"])
def get_latest_signals():
    """Retrieve the latest trading signal per symbol or for a specific symbol."""
    raw_symbol = request.args.get("symbol")
    service = get_signal_service()

    if raw_symbol:
        try:
            clean_sym = validate_symbol(raw_symbol)
        except Exception as exc:
            return api_error(message=f"Invalid symbol format: {exc}", code="INVALID_SYMBOL", status_code=400)
        signals = service.get_latest_signals(symbol=clean_sym)
        return api_response(data=signals.get(clean_sym), meta={"symbol": clean_sym})

    signals = service.get_latest_signals()
    return api_response(data=signals, meta={"count": len(signals)})


@api_bp.route("/signals/history", methods=["GET"])
def get_signal_history():
    """Retrieve paginated trading signal history with optional filtering."""
    raw_symbol = request.args.get("symbol")
    action = request.args.get("action")
    status = request.args.get("status")
    interval = request.args.get("interval")

    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 20))
    except ValueError:
        return api_error(message="page and page_size must be positive integers.", code="INVALID_PAGINATION", status_code=400)

    clean_sym = None
    if raw_symbol:
        try:
            clean_sym = validate_symbol(raw_symbol)
        except Exception as exc:
            return api_error(message=f"Invalid symbol format: {exc}", code="INVALID_SYMBOL", status_code=400)

    service = get_signal_service()
    result = service.get_signal_history(
        symbol=clean_sym,
        action=action,
        status=status,
        interval=interval,
        page=page,
        page_size=page_size,
    )

    return api_response(
        data=result["signals"],
        meta={
            "total": result["total"],
            "page": result["page"],
            "page_size": result["page_size"],
            "total_pages": result["total_pages"],
        },
    )


@api_bp.route("/signals/<id_or_signal_id>", methods=["GET"])
def get_signal_details(id_or_signal_id: str):
    """Retrieve full audit details of a specific trading signal."""
    service = get_signal_service()
    signal_dict = service.get_signal_by_id(id_or_signal_id)
    if not signal_dict:
        return api_error(
            message=f"Trading signal '{id_or_signal_id}' not found.",
            code="NOT_FOUND",
            status_code=404,
        )
    return api_response(data=signal_dict)


@api_bp.route("/signals/health", methods=["GET"])
def get_signals_health():
    """Retrieve signal generation service health, active models, and operational metrics."""
    service = get_signal_service()
    health_data = service.get_service_health()
    return api_response(data=health_data)


@api_bp.route("/signals/generate", methods=["POST"])
def generate_signal():
    """On-demand signal evaluation for a specific symbol and candle bar."""
    body = request.get_json(silent=True) or {}
    raw_symbol = body.get("symbol") or request.args.get("symbol")
    if not raw_symbol:
        return api_error(message="Missing required field 'symbol'.", code="MISSING_FIELD", status_code=400)

    try:
        clean_sym = validate_symbol(raw_symbol)
    except Exception as exc:
        return api_error(message=f"Invalid symbol: {exc}", code="INVALID_SYMBOL", status_code=400)

    interval = str(body.get("interval") or request.args.get("interval") or "1d").strip().lower()
    portfolio_id = body.get("portfolio_id") or request.args.get("portfolio_id")
    candle_input = body.get("candle")

    service = get_signal_service()

    # If candle not explicitly provided, find the latest bar from storage/candle builder
    if not candle_input:
        if interval == "1d":
            # Fetch latest daily bar from historical data
            hist = service.data_service.get_historical_bars(clean_sym, limit=1)
            if hist and hist[0]:
                latest_bar = hist[0][0]
                candle_input = {
                    "symbol": clean_sym,
                    "interval": "1d",
                    "start_time": str(latest_bar.get("timestamp")),
                    "end_time": str(latest_bar.get("timestamp")),
                    "open": float(latest_bar.get("open", 0.0)),
                    "high": float(latest_bar.get("high", 0.0)),
                    "low": float(latest_bar.get("low", 0.0)),
                    "close": float(latest_bar.get("close", 0.0)),
                    "volume": int(latest_bar.get("volume", 0)),
                    "is_complete": True,
                }
        else:
            # Query latest completed candle from MarketCandle table
            from ...models.market_candle import MarketCandle
            mc = (
                MarketCandle.query.filter_by(symbol=clean_sym, interval=interval, is_complete=True)
                .order_by(MarketCandle.start_time.desc())
                .first()
            )
            if mc:
                candle_input = mc.to_dict()
            else:
                try:
                    from ...realtime.manager import RealtimeMarketDataManager
                    rm = RealtimeMarketDataManager.get_instance()
                    latest = rm.get_latest_completed_candle(clean_sym, interval=interval)
                    if not latest:
                        recent_candles = rm.get_candles(clean_sym, interval=interval, limit=1, include_in_progress=False)
                        if recent_candles:
                            latest = recent_candles[-1]
                    if latest:
                        candle_input = latest
                except Exception:
                    pass

    if not candle_input:
        return api_error(
            message=f"No completed candle available for symbol '{clean_sym}' ({interval}) to evaluate.",
            code="NO_CANDLE_AVAILABLE",
            status_code=404,
        )

    signal_res = service.process_completed_candle(
        symbol=clean_sym,
        candle=candle_input,
        interval=interval,
        portfolio_id=portfolio_id,
        data_freshness=body.get("data_freshness", "FRESH"),
    )

    return api_response(data=signal_res)
