import os
from pathlib import Path

from sqlalchemy.pool import StaticPool

# Base directory paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
INSTANCE_DIR = PROJECT_ROOT / "instance"


class Config:
    """Base configuration for TradeMind."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "trademind-dev-secret-key-change-in-production")
    
    # SQLite Database Configuration (stored in instance/)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{INSTANCE_DIR / 'trademind.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {"check_same_thread": False},
    }
    
    # API & Platform Metadata
    APP_NAME = "TradeMind"
    APP_VERSION = "0.1.0"
    PHASE = "Phase 9: Flask API & Database Integration"
    API_PREFIX = "/api/v1"
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
    
    # Background Worker Limits
    MAX_BACKTEST_WORKERS = int(os.environ.get("MAX_BACKTEST_WORKERS", "2"))
    
    # Storage Paths
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Real-Time Market Data Provider Configuration (Phase 11A)
    REALTIME_PROVIDER = os.environ.get("REALTIME_PROVIDER", "kite")
    KITE_API_KEY = os.environ.get("KITE_API_KEY")
    KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN")
    REALTIME_MAX_SUBSCRIPTIONS = int(os.environ.get("REALTIME_MAX_SUBSCRIPTIONS", "50"))
    REALTIME_STALE_TIMEOUT_SECONDS = float(os.environ.get("REALTIME_STALE_TIMEOUT_SECONDS", "15.0"))
    REALTIME_RECONNECT_MAX_RETRIES = int(os.environ.get("REALTIME_RECONNECT_MAX_RETRIES", "5"))


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    ENV = "development"


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = False
    TESTING = True
    ENV = "testing"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False},
    }


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    ENV = "production"
    # Ensure a robust secret key in production
    SECRET_KEY = os.environ.get("SECRET_KEY")
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",") if os.environ.get("CORS_ORIGINS") else []


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
