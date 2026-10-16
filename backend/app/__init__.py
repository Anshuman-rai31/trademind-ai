import os
from pathlib import Path
from flask import Flask

from .config import config_by_name, PROJECT_ROOT, INSTANCE_DIR


def create_app(config_name=None):
    """Application factory for TradeMind Flask backend.
    
    Args:
        config_name (str, optional): Configuration mode ('development', 'testing', 'production').
                                      Defaults to FLASK_ENV or 'development'.
                                      
    Returns:
        Flask: Configured Flask application instance.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development").lower()

    # Locate templates and static directories at project root level
    template_dir = str(PROJECT_ROOT / "templates")
    static_dir = str(PROJECT_ROOT / "static")
    instance_path = str(INSTANCE_DIR)

    # Ensure instance directory exists for SQLite and local storage
    os.makedirs(instance_path, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        instance_path=instance_path,
        instance_relative_config=True,
    )

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Register blueprints
    from .routes.main import main_bp
    from .routes.health import health_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(health_bp)

    return app
