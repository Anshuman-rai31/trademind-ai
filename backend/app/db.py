"""TradeMind Database Configuration and SQLite Persistence Helpers (Phase 9).

Initializes Flask-SQLAlchemy with SQLite foreign key enforcement, robust connection handling,
and safe serialization utilities for NumPy/Pandas trading research artifacts.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()

_app_instance: Optional[Flask] = None


def register_flask_app(app: Flask) -> None:
    """Register active Flask application instance for background worker contexts."""
    global _app_instance
    _app_instance = app


def get_flask_app() -> Optional[Flask]:
    """Retrieve registered Flask application instance."""
    global _app_instance
    return _app_instance


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enforce SQLite foreign key constraints, WAL journaling, and busy timeout."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
    except Exception:
        # Non-SQLite or in-memory dialect fallback
        pass


def safe_serialize(obj: Any) -> Any:
    """Recursively convert NumPy, Pandas, Datetime, and non-native objects to JSON-serializable types.
    
    Guarantees:
        - np.integer -> int
        - np.floating -> float (NaN and Inf converted to None)
        - np.bool_ -> bool
        - np.ndarray -> list
        - pd.Timestamp / datetime / date -> ISO-8601 formatted string
        - pd.Series / pd.DataFrame -> list/dict
        - float NaN / Inf -> None (safe for JSON standard)
    """
    if obj is None:
        return None

    if isinstance(obj, (bool, str, int)):
        return obj

    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (datetime, pd.Timestamp)):
        if obj.tzinfo is None:
            return obj.replace(tzinfo=timezone.utc).isoformat()
        return obj.isoformat()

    if isinstance(obj, (date,)):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return [safe_serialize(x) for x in obj.tolist()]

    if isinstance(obj, pd.Series):
        return {str(k): safe_serialize(v) for k, v in obj.to_dict().items()}

    if isinstance(obj, pd.DataFrame):
        return [safe_serialize(r) for r in obj.to_dict(orient="records")]

    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(x) for x in obj]

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return safe_serialize(obj.to_dict())
        except Exception:
            pass

    return str(obj)


def safe_json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    """Safely serialize any object structure into a valid JSON string."""
    clean_obj = safe_serialize(obj)
    return json.dumps(clean_obj, indent=indent, default=str)


def safe_json_loads(data: Optional[str], default: Any = None) -> Any:
    """Safely parse a JSON string, returning default if null or corrupt."""
    if not data:
        return default
    try:
        return json.loads(data)
    except Exception:
        return default


def seed_default_assets(app: Flask) -> None:
    """Seed canonical supported Indian NSE assets if database table is empty."""
    from .models.asset import Asset

    with app.app_context():
        try:
            if Asset.query.first() is not None:
                return

            default_assets = [
                {
                    "symbol": "RELIANCE.NS",
                    "name": "Reliance Industries Limited",
                    "exchange": "NSE",
                    "asset_type": "EQUITY",
                },
                {
                    "symbol": "TCS.NS",
                    "name": "Tata Consultancy Services Limited",
                    "exchange": "NSE",
                    "asset_type": "EQUITY",
                },
                {
                    "symbol": "INFY.NS",
                    "name": "Infosys Limited",
                    "exchange": "NSE",
                    "asset_type": "EQUITY",
                },
                {
                    "symbol": "ICICIBANK.NS",
                    "name": "ICICI Bank Limited",
                    "exchange": "NSE",
                    "asset_type": "EQUITY",
                },
                {
                    "symbol": "SBIN.NS",
                    "name": "State Bank of India",
                    "exchange": "NSE",
                    "asset_type": "EQUITY",
                },
            ]

            for item in default_assets:
                asset = Asset(
                    symbol=item["symbol"],
                    name=item["name"],
                    exchange=item["exchange"],
                    asset_type=item["asset_type"],
                )
                db.session.add(asset)

            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.warning("Failed to seed default assets: %s", exc)


def seed_default_portfolio(app: Flask) -> None:
    """Seed default simulation/paper trading portfolio if table is empty."""
    from .models.paper_trading import PaperPortfolio

    with app.app_context():
        try:
            if PaperPortfolio.query.filter_by(name="default").first() is not None:
                return
            portfolio = PaperPortfolio(
                name="default",
                initial_cash=100000.0,
                cash=100000.0,
                portfolio_value=100000.0,
                peak_value=100000.0,
            )
            db.session.add(portfolio)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.warning("Failed to seed default simulation portfolio: %s", exc)


def _run_sqlite_column_migrations(app: Flask) -> None:
    """Ensure newly introduced columns exist in SQLite database if created by earlier phases."""
    from sqlalchemy import text
    try:
        with db.engine.connect() as conn:
            # Check paper_portfolios columns
            res = conn.execute(text("PRAGMA table_info(paper_portfolios);")).fetchall()
            existing_cols = {row[1] for row in res} if res else set()

            if existing_cols:
                if "allocation_method" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN allocation_method VARCHAR(30) DEFAULT 'EQUAL_WEIGHT';"))
                if "strategy_type" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN strategy_type VARCHAR(50) DEFAULT 'BUY_AND_HOLD';"))
                if "risk_config_json" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN risk_config_json TEXT;"))
                if "description" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN description TEXT;"))
                if "user_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN user_id VARCHAR(50);"))
                if "is_active" not in existing_cols:
                    conn.execute(text("ALTER TABLE paper_portfolios ADD COLUMN is_active BOOLEAN DEFAULT 1;"))
                conn.commit()

            # Check paper_trades columns
            res_trades = conn.execute(text("PRAGMA table_info(paper_trades);")).fetchall()
            existing_trade_cols = {row[1] for row in res_trades} if res_trades else set()
            if existing_trade_cols and "signal_id" not in existing_trade_cols:
                conn.execute(text("ALTER TABLE paper_trades ADD COLUMN signal_id VARCHAR(40);"))
                conn.commit()
    except Exception as exc:
        app.logger.warning("Migration check error (non-fatal): %s", exc)


def init_db(app: Flask) -> None:
    """Initialize database tables and seed baseline reference assets and default portfolio."""
    # Explicitly import all models so SQLAlchemy metadata registers all tables
    from .models.agent import AgentRun  # noqa: F401
    from .models.asset import Asset  # noqa: F401
    from .models.backtest import BacktestRun, PortfolioSnapshotRecord, TradeRecord  # noqa: F401
    from .models.market_candle import MarketCandle  # noqa: F401
    from .models.model_registry import RegisteredModel  # noqa: F401
    from .models.paper_trading import (  # noqa: F401
        PaperPortfolio,
        PaperPosition,
        PaperSnapshotRecord,
        PaperTrade,
        PortfolioBucketItem,
        RiskEventRecord,
    )
    from .models.trading_signal import RLTradingSignal  # noqa: F401

    with app.app_context():
        db.create_all()
        _run_sqlite_column_migrations(app)
        seed_default_assets(app)
        seed_default_portfolio(app)
