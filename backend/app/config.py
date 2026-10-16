import os
from pathlib import Path

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
    
    # Platform Metadata
    APP_NAME = "TradeMind"
    APP_VERSION = "0.1.0"
    PHASE = "Phase 1: Project Setup"
    
    # Storage Paths
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


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


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    ENV = "production"
    # Ensure a robust secret key in production
    SECRET_KEY = os.environ.get("SECRET_KEY")


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
