"""Real-Time Market Data REST API Routes (Phase 11A / 11B).

Provides endpoints for monitoring connection health, managing portfolio-aware
subscriptions, triggering reconnections with reconciliation, and retrieving
validated streaming quotes without exposing credentials.
"""
from __future__ import annotations

from flask import current_app, request

from ...data.validator import validate_symbol
from ...realtime.manager import RealtimeMarketDataManager
from ...realtime.models import ProviderConfig
from ...services.trading_pipeline_service import TradingPipelineService
from . import api_bp, api_error, api_response


def get_realtime_manager() -> RealtimeMarketDataManager:
    """Helper to retrieve or initialize the real-time manager with application config."""
    cfg = ProviderConfig(
        provider_name=current_app.config.get("REALTIME_PROVIDER", "kite"),
        api_key=current_app.config.get("KITE_API_KEY"),
        access_token=current_app.config.get("KITE_ACCESS_TOKEN"),
        max_subscriptions=current_app.config.get("REALTIME_MAX_SUBSCRIPTIONS", 50),
        stale_timeout_seconds=current_app.config.get("REALTIME_STALE_TIMEOUT_SECONDS", 15.0),
        reconnect_max_retries=current_app.config.get("REALTIME_RECONNECT_MAX_RETRIES", 5),
    )
    return RealtimeMarketDataManager.get_instance(config=cfg)


@api_bp.route("/realtime/status", methods=["GET"])
def get_realtime_status():
    """Retrieve real-time market data connection status and feed telemetry.
    
    Security:
        Credentials and tokens are strictly excluded from output.
    """
    manager = get_realtime_manager()
    telemetry = manager.get_telemetry()
    safe_config = manager.config.to_safe_dict()

    return api_response(data={
        "status": telemetry.status.value,
        "provider": telemetry.provider_name,
        "is_stale": telemetry.is_stale,
        "telemetry": telemetry.to_dict(),
        "config": safe_config,
    })


@api_bp.route("/realtime/health", methods=["GET"])
def get_realtime_health():
    """Retrieve granular real-time health diagnostic distinguishing socket state from inactive instruments."""
    manager = get_realtime_manager()
    report = manager.get_health_report()
    return api_response(data=report)


@api_bp.route("/realtime/connect", methods=["POST"])
def connect_realtime():
    """Initiate connection to the real-time market data provider."""
    manager = get_realtime_manager()
    success = manager.connect()
    telemetry = manager.get_telemetry()

    if not success:
        return api_error(
            message=telemetry.last_error or "Failed to connect to real-time market data provider.",
            code="CONNECTION_FAILED",
            status_code=503 if telemetry.status.value == "AUTH_FAILED" else 500,
            details=telemetry.to_dict(),
        )

    return api_response(data={
        "connected": True,
        "status": telemetry.status.value,
        "telemetry": telemetry.to_dict(),
    })


@api_bp.route("/realtime/disconnect", methods=["POST"])
def disconnect_realtime():
    """Cleanly disconnect from the real-time market data provider."""
    manager = get_realtime_manager()
    manager.disconnect()
    telemetry = manager.get_telemetry()

    return api_response(data={
        "connected": False,
        "status": telemetry.status.value,
    })


@api_bp.route("/realtime/reconnect", methods=["POST"])
def reconnect_realtime():
    """Trigger clean reconnection with automatic subscription reconciliation."""
    manager = get_realtime_manager()
    success = manager.reconnect()
    telemetry = manager.get_telemetry()

    if not success:
        return api_error(
            message=telemetry.last_error or "Reconnection failed.",
            code="RECONNECT_FAILED",
            status_code=500,
            details=telemetry.to_dict(),
        )

    return api_response(data={
        "reconnected": True,
        "status": telemetry.status.value,
        "subscriptions": manager.get_subscribed_symbols(),
    })


@api_bp.route("/realtime/subscriptions", methods=["GET"])
def get_subscriptions():
    """List currently subscribed instruments with optional detailed state breakdown."""
    manager = get_realtime_manager()
    include_details = request.args.get("details", "").lower() in ("true", "1", "yes")

    if include_details:
        details = manager.get_subscription_details()
        return api_response(data=details)

    symbols = manager.get_subscribed_symbols()
    return api_response(data={
        "count": len(symbols),
        "symbols": symbols,
        "max_subscriptions": manager.config.max_subscriptions,
    })


@api_bp.route("/realtime/subscribe", methods=["POST"])
def subscribe_instruments():
    """Subscribe to real-time market updates for a list of instruments."""
    payload = request.get_json(silent=True) or {}
    symbols = payload.get("symbols", [])
    if not symbols or not isinstance(symbols, list):
        return api_error(message="Field 'symbols' must be a non-empty list of symbols.", status_code=400)

    manager = get_realtime_manager()
    ok = manager.subscribe(symbols)
    if not ok:
        telemetry = manager.get_telemetry()
        return api_error(
            message=telemetry.last_error or "Failed to subscribe to instruments.",
            code="SUBSCRIPTION_ERROR",
            status_code=400,
        )

    return api_response(data={
        "subscribed": True,
        "current_subscriptions": manager.get_subscribed_symbols(),
    })


@api_bp.route("/realtime/unsubscribe", methods=["POST"])
def unsubscribe_instruments():
    """Unsubscribe from real-time market updates for a list of instruments."""
    payload = request.get_json(silent=True) or {}
    symbols = payload.get("symbols", [])
    if not symbols or not isinstance(symbols, list):
        return api_error(message="Field 'symbols' must be a non-empty list of symbols.", status_code=400)

    manager = get_realtime_manager()
    manager.unsubscribe(symbols)

    return api_response(data={
        "unsubscribed": True,
        "current_subscriptions": manager.get_subscribed_symbols(),
    })


@api_bp.route("/realtime/sync-portfolio", methods=["POST"])
def sync_portfolio_subscriptions():
    """Synchronize real-time market data subscriptions to match a portfolio's stock bucket."""
    payload = request.get_json(silent=True) or {}
    portfolio_name = payload.get("portfolio") or payload.get("name") or "default"

    trading_service = TradingPipelineService()
    portfolio = trading_service.get_portfolio(portfolio_name)
    if not portfolio:
        return api_error(message=f"Portfolio '{portfolio_name}' not found.", code="NOT_FOUND", status_code=404)

    bucket_symbols = [item.symbol for item in portfolio.bucket_items]
    if not bucket_symbols:
        bucket_symbols = trading_service.get_supported_symbols()

    manager = get_realtime_manager()
    result = manager.sync_portfolio_subscriptions(
        portfolio_symbols=bucket_symbols,
        portfolio_name=portfolio.name,
    )

    return api_response(data=result)


@api_bp.route("/realtime/quotes", methods=["GET"])
def get_all_quotes():
    """Retrieve all latest validated quotes for subscribed instruments."""
    manager = get_realtime_manager()
    quotes = manager.get_all_quotes()
    return api_response(data={
        "count": len(quotes),
        "quotes": {sym: q.to_dict() for sym, q in quotes.items()},
        "feed_status": manager.get_status().value,
    })


@api_bp.route("/realtime/quotes/<symbol>", methods=["GET"])
def get_quote_for_symbol(symbol: str):
    """Retrieve latest validated real-time quote for a specific instrument."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    manager = get_realtime_manager()
    quote = manager.get_latest_quote(clean_sym)
    if not quote:
        return api_error(
            message=f"No real-time quote available for '{clean_sym}'. Ensure symbol is subscribed and feed is connected.",
            code="QUOTE_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=quote.to_dict())


@api_bp.route("/realtime/freshness/<symbol>", methods=["GET"])
def get_freshness_for_symbol(symbol: str):
    """Retrieve data freshness, latency ms, and event timestamps for an instrument."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    manager = get_realtime_manager()
    freshness = manager.get_symbol_freshness(clean_sym)
    return api_response(data=freshness)


@api_bp.route("/realtime/candles/<symbol>", methods=["GET"])
def get_candles_for_symbol(symbol: str):
    """Retrieve recent completed and in-progress candles for an instrument."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    interval = request.args.get("interval", "5m").lower()
    if interval not in ("1m", "5m", "15m", "1d"):
        interval = "5m"

    try:
        limit = min(500, max(1, int(request.args.get("limit", 50))))
    except (ValueError, TypeError):
        limit = 50

    include_in_progress = request.args.get("include_in_progress", "true").lower() in ("true", "1", "yes")

    manager = get_realtime_manager()
    candles = manager.get_candles(
        symbol=clean_sym,
        interval=interval,
        limit=limit,
        include_in_progress=include_in_progress,
    )

    return api_response(data={
        "symbol": clean_sym,
        "interval": interval,
        "count": len(candles),
        "candles": candles,
    })


@api_bp.route("/realtime/candles/<symbol>/latest", methods=["GET"])
def get_latest_completed_candle(symbol: str):
    """Retrieve the latest closed/completed candle for an instrument."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    interval = request.args.get("interval", "5m").lower()
    manager = get_realtime_manager()
    candle = manager.get_latest_completed_candle(clean_sym, interval=interval)

    if not candle:
        return api_error(
            message=f"No completed {interval} candle available for '{clean_sym}'.",
            code="CANDLE_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=candle)


@api_bp.route("/realtime/candles/<symbol>/in-progress", methods=["GET"])
def get_in_progress_candle(symbol: str):
    """Retrieve the currently accumulating in-progress candle for an instrument."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    interval = request.args.get("interval", "5m").lower()
    manager = get_realtime_manager()
    candle = manager.get_in_progress_candle(clean_sym, interval=interval)

    if not candle:
        return api_error(
            message=f"No in-progress {interval} candle currently accumulating for '{clean_sym}'.",
            code="CANDLE_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=candle)


@api_bp.route("/realtime/features/<symbol>", methods=["GET"])
def get_live_features(symbol: str):
    """Evaluate PPO feature engineering readiness and timeframe compatibility."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    interval = request.args.get("interval", "5m").lower()
    manager = get_realtime_manager()
    readiness = manager.get_feature_readiness(clean_sym, interval=interval)
    return api_response(data=readiness)


@api_bp.route("/realtime/dashboard", methods=["GET"])
def get_realtime_dashboard():
    """Retrieve unified real-time dashboard state (quotes, candles, freshness, features)."""
    symbol = request.args.get("symbol", "RELIANCE.NS")
    try:
        clean_sym = validate_symbol(symbol)
    except Exception:
        clean_sym = str(symbol).strip().upper()

    interval = request.args.get("interval", "5m").lower()
    portfolio_name = request.args.get("portfolio", "default")

    manager = get_realtime_manager()
    payload = manager.get_dashboard_payload(
        symbol=clean_sym,
        portfolio_name=portfolio_name,
        interval=interval,
    )
    return api_response(data=payload)

