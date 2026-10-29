"""Model Registry REST API Routes for TradeMind (Phase 9.5B).

Provides endpoints for listing registered models, inspecting artifact metadata,
approving models for inference eligibility, activating models, and rolling back.
"""
from __future__ import annotations

from flask import request

from ...data.validator import validate_symbol
from ...services.model_registry_service import ModelRegistryService
from . import api_bp, api_error, api_response


@api_bp.route("/models", methods=["GET"])
def list_models():
    """List registered models with optional symbol and status filtering."""
    symbol = request.args.get("symbol")
    status = request.args.get("status")
    service = ModelRegistryService()
    # Auto-sync any new checkpoints from disk
    service.sync_local_artifacts()
    models = service.list_models(symbol=symbol, status=status)
    return api_response(data=models, meta={"count": len(models)})


@api_bp.route("/models/<id_or_model_id>", methods=["GET"])
def get_model(id_or_model_id: str):
    """Retrieve detailed metadata and checksum for a single registered model."""
    service = ModelRegistryService()
    model = service.get_model(id_or_model_id)
    if not model:
        return api_error(
            message=f"Model '{id_or_model_id}' not found in registry.",
            code="NOT_FOUND",
            status_code=404,
        )
    return api_response(data=model.to_dict())


@api_bp.route("/models/<id_or_model_id>/approve", methods=["POST"])
def approve_model(id_or_model_id: str):
    """Promote a candidate model to approved status."""
    service = ModelRegistryService()
    try:
        model = service.approve_model(id_or_model_id)
        return api_response(
            data=model.to_dict(),
            meta={"message": f"Model '{model.model_id}' promoted to approved status."},
        )
    except FileNotFoundError as exc:
        return api_error(message=str(exc), code="ARTIFACT_MISSING", status_code=404)
    except ValueError as exc:
        return api_error(message=str(exc), code="BAD_REQUEST", status_code=400)
    except Exception as exc:
        return api_error(message=f"Error approving model: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/models/<id_or_model_id>/activate", methods=["POST"])
def activate_model(id_or_model_id: str):
    """Set model as the active inference policy for its symbol."""
    service = ModelRegistryService()
    try:
        model = service.activate_model(id_or_model_id)
        return api_response(
            data=model.to_dict(),
            meta={"message": f"Model '{model.model_id}' activated for {model.symbol}."},
        )
    except FileNotFoundError as exc:
        return api_error(message=str(exc), code="ARTIFACT_MISSING", status_code=404)
    except ValueError as exc:
        return api_error(message=str(exc), code="BAD_REQUEST", status_code=400)
    except Exception as exc:
        return api_error(message=f"Error activating model: {exc}", code="SERVER_ERROR", status_code=500)


@api_bp.route("/models/<symbol>/rollback", methods=["POST"])
def rollback_model(symbol: str):
    """Roll back active status for a symbol to the previous approved model."""
    service = ModelRegistryService()
    try:
        rolled_back = service.rollback_model(symbol)
        if rolled_back:
            return api_response(
                data=rolled_back.to_dict(),
                meta={"message": f"Rolled back {symbol} to model '{rolled_back.model_id}'."},
            )
        return api_response(
            data=None,
            meta={"message": f"Active model deactivated for {symbol}; no prior approved model found."},
        )
    except Exception as exc:
        return api_error(message=f"Error rolling back model: {exc}", code="SERVER_ERROR", status_code=500)
