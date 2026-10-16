from flask import Blueprint, render_template, current_app

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET"])
def index():
    """Renders the TradeMind landing page."""
    context = {
        "app_name": current_app.config.get("APP_NAME", "TradeMind"),
        "version": current_app.config.get("APP_VERSION", "0.1.0"),
        "phase": current_app.config.get("PHASE", "Phase 1: Project Setup"),
        "env": current_app.config.get("ENV", "development"),
    }
    return render_template("index.html", **context)
