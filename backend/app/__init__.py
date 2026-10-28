"""TradeMind Flask Application Factory (Phase 9 Integration).

Configures SQLite database persistence via Flask-SQLAlchemy, CORS middleware,
institutional REST API blueprints, and centralized error handling.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import INSTANCE_DIR, PROJECT_ROOT, config_by_name
from .db import db, init_db


def create_app(config_name: Optional[str] = None) -> Flask:
    """Application factory for TradeMind Flask backend.
    
    Args:
        config_name: Configuration mode ('development', 'testing', 'production').
                     Defaults to FLASK_ENV or 'development'.
                     
    Returns:
        Flask: Fully configured Flask application instance.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development").lower()

    # Locate templates and static directories at project root level
    template_dir = str(PROJECT_ROOT / "templates")
    static_dir = str(PROJECT_ROOT / "static")
    instance_path = str(INSTANCE_DIR)

    # Ensure instance directory exists for SQLite storage
    os.makedirs(instance_path, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        instance_path=instance_path,
        instance_relative_config=True,
    )

    # 1. Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # 2. Initialize Database Persistence
    db.init_app(app)
    init_db(app)

    # 3. Configure CORS (Cross-Origin Resource Sharing)
    cors_origins = app.config.get("CORS_ORIGINS", "*")
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})

    # 4. Register Blueprints
    from .routes.main import main_bp
    from .routes.health import health_bp
    from .routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp)

    # 5. Global API Error Handlers
    @app.errorhandler(404)
    def handle_not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Resource not found: {request.path}",
                },
                "meta": {
                    "timestamp": os.environ.get("CURRENT_TIME", "") or None,
                },
            }), 404
        return "Not Found", 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": {
                    "code": "METHOD_NOT_ALLOWED",
                    "message": f"Method {request.method} is not allowed for {request.path}",
                },
            }), 405
        return "Method Not Allowed", 405

    @app.errorhandler(500)
    def handle_server_error(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing the request.",
                },
            }), 500
        return "Internal Server Error", 500

    return app
