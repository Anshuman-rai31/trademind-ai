"""Market Data Service for TradeMind (Phase 9).

Encapsulates data ingestion lookup, validation, historical OHLCV queries,
and symbol inventory inspection without downloading large external sets on every request.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import PROJECT_ROOT
from ..data.pipeline import MarketDataPipeline, sanitize_filename_component
from ..data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_SYMBOL, COL_TIMESTAMP, COL_VOLUME
from ..data.validator import validate_symbol
from ..db import safe_serialize
from ..models.asset import Asset


class MarketDataService:
    """Service layer interfacing with MarketDataPipeline and SQLite Asset repository."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = Path(processed_dir) if processed_dir else PROJECT_ROOT / "data" / "processed"
        self.pipeline = MarketDataPipeline(processed_dir=self.processed_dir)

    def list_assets(self) -> List[Dict[str, Any]]:
        """List all canonical assets persisted in SQLite."""
        assets = Asset.query.order_by(Asset.symbol.asc()).all()
        return [a.to_dict() for a in assets]

    def get_asset_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve details and data provenance summary for a single asset."""
        try:
            clean_sym = validate_symbol(symbol)
        except Exception:
            return None

        asset = Asset.query.filter_by(symbol=clean_sym).first()
        if not asset:
            return None

        data_summary = self._inspect_processed_data_summary(clean_sym)
        res = asset.to_dict()
        res["market_data_summary"] = data_summary
        return res

    def get_historical_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        """Fetch validated historical OHLCV records for a symbol from data/processed.
        
        Returns:
            Tuple of (records_list, total_matched_count), or None if no dataset exists.
        """
        clean_sym = validate_symbol(symbol)
        data_path = self._find_processed_file(clean_sym)
        if not data_path or not data_path.exists():
            return None

        try:
            df = pd.read_csv(data_path)
        except Exception:
            return None

        if df.empty or COL_TIMESTAMP not in df.columns:
            return None

        # Standardize timestamp parsing
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=True)
        df = df.sort_values(by=COL_TIMESTAMP)

        # Filter by date range (inclusive)
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date, utc=True)
                df = df[df[COL_TIMESTAMP] >= start_dt]
            except Exception:
                raise ValueError(f"Invalid start_date format: {start_date}")

        if end_date:
            try:
                # Include full end-of-day by setting 23:59:59 if no time component
                end_dt = pd.to_datetime(end_date, utc=True)
                if len(end_date.strip()) == 10:
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)
                df = df[df[COL_TIMESTAMP] <= end_dt]
            except Exception:
                raise ValueError(f"Invalid end_date format: {end_date}")

        total_matched = len(df)
        if limit and limit > 0:
            df = df.iloc[:limit]

        # Format records for JSON response
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            record = {
                "timestamp": row[COL_TIMESTAMP].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": safe_serialize(row.get(COL_OPEN)),
                "high": safe_serialize(row.get(COL_HIGH)),
                "low": safe_serialize(row.get(COL_LOW)),
                "close": safe_serialize(row.get(COL_CLOSE)),
                "volume": safe_serialize(row.get(COL_VOLUME)),
                "symbol": clean_sym,
            }
            records.append(record)

        return records, total_matched

    def get_historical_dataframe(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Load historical data as a pandas DataFrame for simulation and ML engines."""
        clean_sym = validate_symbol(symbol)
        data_path = self._find_processed_file(clean_sym)
        if not data_path or not data_path.exists():
            return None

        try:
            df = pd.read_csv(data_path)
            if df.empty or COL_TIMESTAMP not in df.columns:
                return None
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=True)
            df = df.sort_values(by=COL_TIMESTAMP).reset_index(drop=True)

            if start_date:
                start_dt = pd.to_datetime(start_date, utc=True)
                df = df[df[COL_TIMESTAMP] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date, utc=True)
                if len(end_date.strip()) == 10:
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)
                df = df[df[COL_TIMESTAMP] <= end_dt]

            return df.reset_index(drop=True)
        except Exception:
            return None

    def _find_processed_file(self, symbol: str) -> Optional[Path]:
        """Locate the processed CSV file for a given symbol."""
        # 1. Canonical naming convention: {symbol}_1d_processed.csv
        canonical = self.pipeline.get_canonical_processed_path(symbol, "1d")
        if canonical.exists():
            return canonical

        # 2. Date-tagged or alternate file matching
        safe_sym = sanitize_filename_component(symbol)
        pattern = f"{safe_sym}_*.csv"
        matching = sorted(self.processed_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching:
            return matching[0]

        # 3. Direct match
        direct = self.processed_dir / f"{safe_sym}.csv"
        if direct.exists():
            return direct

        return None

    def _inspect_processed_data_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Extract lightweight metadata summary of processed data on disk."""
        data_path = self._find_processed_file(symbol)
        if not data_path:
            return None

        try:
            df = pd.read_csv(data_path)
            if df.empty or COL_TIMESTAMP not in df.columns:
                return None
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=True)
            return {
                "file_name": data_path.name,
                "total_bars": len(df),
                "start_date": df[COL_TIMESTAMP].min().strftime("%Y-%m-%d"),
                "end_date": df[COL_TIMESTAMP].max().strftime("%Y-%m-%d"),
                "has_volume": COL_VOLUME in df.columns,
            }
        except Exception:
            return None
