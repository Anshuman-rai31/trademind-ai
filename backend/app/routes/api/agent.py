"""PPO Agent REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

from flask import request

from ...data.validator import validate_symbol
from ...services.agent_service import AgentService, ModelUnavailableError
from . import api_bp, api_error, api_response


@api_bp.route("/agent/status", methods=["GET"])
def get_agent_status():
    """Return PPO agent installation status, model artifact availability, and readiness."""
    symbol = request.args.get("symbol")
    agent_service = AgentService()
    status = agent_service.get_agent_status(symbol=symbol)
    return api_response(data=status, status_code=200)


@api_bp.route("/agent/metrics", methods=["GET"])
def get_agent_metrics():
    """Return actual persisted evaluation metrics for the trained PPO agent."""
    symbol = request.args.get("symbol")
    agent_service = AgentService()
    metrics_info = agent_service.get_persisted_metrics(symbol=symbol)
    if not metrics_info:
        return api_response(
            data=None,
            meta={"message": "No persisted agent evaluation metrics available."},
            status_code=200,
        )

    return api_response(data=metrics_info, status_code=200)


@api_bp.route("/agent/decisions", methods=["POST"])
def get_agent_decision():
    """Provide a deterministic inference decision from a trained PPO policy model.
    
    Guarantees:
        - Never triggers model training automatically.
        - Returns 503 Service Unavailable if no trained model artifact is found on disk.
        - Clearly indicates model output is for quantitative research and simulation only.
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

    state = payload.get("state") or payload.get("features")
    if state is None:
        return api_error(
            message="Field 'state' or 'features' is required (list of floats or feature dictionary).",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    agent_service = AgentService()
    try:
        decision = agent_service.predict_action(clean_sym, state)
        return api_response(data=decision, status_code=200)
    except ModelUnavailableError as exc:
        return api_error(
            message=str(exc),
            code="MODEL_UNAVAILABLE",
            status_code=503,
        )
    except ValueError as exc:
        return api_error(
            message=str(exc),
            code="INVALID_OBSERVATION",
            status_code=400,
        )
    except Exception as exc:
        return api_error(
            message=f"Error evaluating PPO decision: {exc}",
            code="SERVER_ERROR",
            status_code=500,
        )
