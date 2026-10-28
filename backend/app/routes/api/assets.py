"""Asset Management REST API Routes for TradeMind (Phase 9)."""

from __future__ import annotations

import re
from typing import Optional

from flask import request

from ...data.validator import validate_symbol
from ...services.market_data_service import MarketDataService
from . import api_bp, api_error, api_response


@api_bp.route("/assets", methods=["GET"])
def list_assets():
    """List all supported assets configured in SQLite persistence."""
    data_service = MarketDataService()
    assets = data_service.list_assets()
    return api_response(data={"assets": assets, "total": len(assets)}, status_code=200)


@api_bp.route("/assets/<symbol>", methods=["GET"])
def get_asset_details(symbol: str):
    """Retrieve details, exchange classification, and data summary for a single asset."""
    data_service = MarketDataService()
    asset = data_service.get_asset_details(symbol)
    if not asset:
        return api_error(
            message=f"Asset with symbol '{symbol}' not found.",
            code="ASSET_NOT_FOUND",
            status_code=404,
        )

    return api_response(data=asset, status_code=200)


@api_bp.route("/assets/<symbol>/history", methods=["GET"])
def get_asset_history(symbol: str):
    """Query validated historical OHLCV data from processed dataset."""
    try:
        clean_sym = validate_symbol(symbol)
    except Exception as exc:
        return api_error(
            message=f"Invalid symbol format '{symbol}': {exc}",
            code="INVALID_SYMBOL",
            status_code=400,
        )

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit_str = request.args.get("limit", "1000")

    # Validate limit
    try:
        limit = int(limit_str)
        if limit <= 0 or limit > 5000:
            return api_error(
                message="Query parameter 'limit' must be an integer between 1 and 5000.",
                code="INVALID_PARAMETER",
                status_code=400,
            )
    except ValueError:
        return api_error(
            message="Query parameter 'limit' must be a valid integer.",
            code="INVALID_PARAMETER",
            status_code=400,
        )

    # Validate date formats
    date_regex = r"^\d{4}-\d{2}-\d{2}$"
    if start_date:
        if not re.match(date_regex, start_date.strip()):
            return api_error(
                message="Parameter 'start_date' must be in YYYY-MM-DD format.",
                code="INVALID_DATE_FORMAT",
                status_code=400,
            )
    if end_date:
        if not re.match(date_regex, end_date.strip()):
            return api_error(
                message="Parameter 'end_date' must be in YYYY-MM-DD format.",
                code="INVALID_DATE_FORMAT",
                status_code=400,
            )

    if start_date and end_date and start_date > end_date:
        return api_error(
            message=f"start_date ({start_date}) must be earlier than or equal to end_date ({end_date}).",
            code="INVALID_DATE_RANGE",
            status_code=400,
        )

    data_service = MarketDataService()
    try:
        result = data_service.get_historical_bars(
            clean_sym,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except ValueError as exc:
        return api_error(message=str(exc), code="INVALID_QUERY", status_code=400)

    if result is None:
        return api_error(
            message=f"Historical data is not available for symbol '{clean_sym}'.",
            code="DATA_UNAVAILABLE",
            status_code=404,
        )

    bars, total_matched = result
    meta = {
        "symbol": clean_sym,
        "returned_bars": len(bars),
        "total_matched_bars": total_matched,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    return api_response(data=bars, meta=meta, status_code=200)
