"""Route package initialization."""
from .main import main_bp
from .health import health_bp

__all__ = ["main_bp", "health_bp"]
