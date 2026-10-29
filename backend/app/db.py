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


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enforce SQLite foreign key constraints and journaling performance."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
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

    with app.app_context():
        db.create_all()
        seed_default_assets(app)
        seed_default_portfolio(app)
