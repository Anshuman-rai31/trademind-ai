"""Risk Management REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

from flask import request

from ...data.validator import validate_symbol
from ...services.risk_service import RiskService
from . import api_bp, api_error, api_response


@api_bp.route("/risk/config", methods=["GET"])
def get_risk_config():
    """Return active quantitative risk configuration parameters."""
    risk_service = RiskService()
    config = risk_service.get_config()
    return api_response(data=config, status_code=200)


@api_bp.route("/risk/status", methods=["GET"])
def get_risk_status():
    """Return available risk state, active constraints, and operational readiness."""
    risk_service = RiskService()
    status = risk_service.get_status()
    return api_response(data=status, status_code=200)


@api_bp.route("/risk/evaluate", methods=["POST"])
def evaluate_risk():
    """Evaluate a proposed trade order and portfolio state through the RiskManager gateway.
    
    Returns:
        - Requested action
        - Effective action (approved quantity, side)
        - Whether intervention occurred
        - Intervention reason and details
    """
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return api_error(
            message="Request body must be a valid JSON object.",
            code="MALFORMED_JSON",
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

    side = payload.get("side")
    if not side or not isinstance(side, str) or side.strip().upper() not in ("BUY", "SELL"):
        return api_error(
            message="Field 'side' is required and must be 'BUY' or 'SELL'.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    raw_qty = payload.get("quantity")
    try:
        quantity = float(raw_qty)
        if quantity <= 0:
            return api_error(
                message="Field 'quantity' must be strictly positive.",
                code="INVALID_PARAMETER",
                status_code=400,
            )
    except (ValueError, TypeError):
        return api_error(
            message="Field 'quantity' must be a valid positive number.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    raw_price = payload.get("price")
    try:
        price = float(raw_price)
        if price <= 0:
            return api_error(
                message="Field 'price' must be strictly positive.",
                code="INVALID_PARAMETER",
                status_code=400,
            )
    except (ValueError, TypeError):
        return api_error(
            message="Field 'price' must be a valid positive number.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    portfolio_state = payload.get("portfolio", {})
    if not isinstance(portfolio_state, dict):
        return api_error(
            message="Optional field 'portfolio' must be a dictionary.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    risk_service = RiskService()
    try:
        evaluation = risk_service.evaluate_proposed_action(
            symbol=clean_sym,
            side=side,
            quantity=quantity,
            price=price,
            portfolio_state=portfolio_state,
        )
        return api_response(data=evaluation, status_code=200)
    except ValueError as exc:
        return api_error(message=str(exc), code="INVALID_RISK_EVALUATION", status_code=400)
    except Exception as exc:
        return api_error(
            message=f"Error evaluating risk constraints: {exc}",
            code="SERVER_ERROR",
            status_code=500,
        )
