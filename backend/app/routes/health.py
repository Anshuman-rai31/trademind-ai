from datetime import datetime, timezone
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
@health_bp.route("/api/health", methods=["GET"])
def check_health():
    """Health check endpoint confirming service status and environment.
    
    Returns:
        JSON response with system health, timestamp, and platform metadata.
    """
    return jsonify({
        "status": "healthy",
        "service": current_app.config.get("APP_NAME", "TradeMind"),
        "version": current_app.config.get("APP_VERSION", "0.1.0"),
        "phase": current_app.config.get("PHASE", "Phase 1: Project Setup"),
        "environment": current_app.config.get("ENV", "development"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modules": {
            "web_server": "operational",
            "sqlite_storage": "ready",
            "routes": "registered",
        }
    }), 200
