"""TradeMind REST API Blueprint and Response Utilities (Phase 9).

Standardizes JSON envelopes, error formatting, UTC metadata, and blueprint route registration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, Response, jsonify, request

from ...db import safe_serialize

api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def api_response(
    data: Any = None,
    status_code: int = 200,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Response, int]:
    """Build a standardized successful API response envelope."""
    payload = {
        "success": True,
        "data": safe_serialize(data),
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        },
    }
    return jsonify(payload), status_code


def api_error(
    message: str,
    code: str = "BAD_REQUEST",
    status_code: int = 400,
    details: Any = None,
) -> Tuple[Response, int]:
    """Build a standardized error response envelope."""
    error_dict: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details is not None:
        error_dict["details"] = safe_serialize(details)

    payload = {
        "success": False,
        "error": error_dict,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return jsonify(payload), status_code


# Import sub-modules to register endpoints onto api_bp
from . import agent, assets, backtests, dashboard, health, models, realtime, regime, risk, signals, trading  # noqa: E402, F401


