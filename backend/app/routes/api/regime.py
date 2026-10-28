"""Market-Regime Detection REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

from ...data.validator import validate_symbol
from ...services.regime_service import RegimeService
from . import api_bp, api_error, api_response


@api_bp.route("/regimes/<symbol>", methods=["GET"])
def get_regime_info(symbol: str):
    """Return available market-regime detector configuration and model metadata for a symbol."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception as exc:
        return api_error(
            message=f"Invalid symbol '{symbol}': {exc}",
            code="INVALID_SYMBOL",
            status_code=400,
        )

    regime_service = RegimeService()
    info = regime_service.get_regime_info(clean_sym)
    if not info:
        return api_error(
            message=f"No fitted market-regime detector model found for symbol '{clean_sym}'.",
            code="REGIME_MODEL_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=info, status_code=200)


@api_bp.route("/regimes/<symbol>/summary", methods=["GET"])
def get_regime_summary(symbol: str):
    """Return available regime cluster summaries, descriptive labels, and empirical statistics."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception as exc:
        return api_error(
            message=f"Invalid symbol '{symbol}': {exc}",
            code="INVALID_SYMBOL",
            status_code=400,
        )

    regime_service = RegimeService()
    summaries = regime_service.get_cluster_summaries(clean_sym)
    if not summaries:
        return api_error(
            message=f"No cluster summaries available for symbol '{clean_sym}'. Model artifact missing.",
            code="REGIME_MODEL_NOT_FOUND",
            status_code=404,
        )

    meta = {
        "symbol": clean_sym,
        "cluster_count": len(summaries),
    }
    return api_response(data=summaries, meta=meta, status_code=200)


@api_bp.route("/regimes/<symbol>/latest", methods=["GET"])
def get_latest_regime(symbol: str):
    """Return retrospective regime classification for the latest validated bar.
    
    Guarantees:
        - Does NOT retrain the detector automatically.
        - Clearly distinguishes historical classification from real-time price prediction.
    """
    try:
        clean_sym = validate_symbol(symbol)
    except Exception as exc:
        return api_error(
            message=f"Invalid symbol '{symbol}': {exc}",
            code="INVALID_SYMBOL",
            status_code=400,
        )

    regime_service = RegimeService()
    model_path = regime_service.find_model_path(clean_sym)
    if not model_path:
        return api_error(
            message=f"Fitted market-regime detector artifact not found for symbol '{clean_sym}'.",
            code="REGIME_MODEL_NOT_FOUND",
            status_code=404,
        )

    latest_info = regime_service.get_latest_regime(clean_sym)
    if not latest_info:
        return api_error(
            message=f"Insufficient historical data available to compute regime features for '{clean_sym}'.",
            code="DATA_UNAVAILABLE",
            status_code=404,
        )

    return api_response(data=latest_info, status_code=200)
