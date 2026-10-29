"""Market Data Service for TradeMind (Phase 9).

Encapsulates data ingestion lookup, validation, historical OHLCV queries,
and symbol inventory inspection without downloading large external sets on every request.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)
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
        include_indicators: bool = False,
    ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        """Fetch validated historical OHLCV records for a symbol from data/processed.
        
        Args:
            symbol: Canonical asset ticker (e.g. 'RELIANCE.NS').
            start_date: Optional lower bound (inclusive, YYYY-MM-DD).
            end_date: Optional upper bound (inclusive, YYYY-MM-DD).
            limit: Maximum number of rows to return (default: 1000).
            include_indicators: If True, computes SMA(5), SMA(20), RSI(14), and MACD(12,26,9)
                                on the full chronological series before slicing, preventing
                                future-data leakage while preserving warmup.
        
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

        # Standardize timestamp parsing and chronological order
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=True)
        df = df.sort_values(by=COL_TIMESTAMP)

        # Compute technical indicators on the full series to prevent future leakage
        # and guarantee complete indicator warm-up across date filters
        if include_indicators and not df.empty:
            try:
                from ..features.engineer import compute_moving_averages, compute_rsi, compute_macd
                df = compute_moving_averages(df, sma_windows=[5, 20])
                df = compute_rsi(df, period=14)
                df = compute_macd(df, fast=12, slow=26, signal=9)
            except Exception as exc:
                logger.warning("Failed to compute technical indicators for %s: %s", clean_sym, exc)

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
            if include_indicators:
                record["sma_5"] = safe_serialize(row.get("sma_5"))
                record["sma_20"] = safe_serialize(row.get("sma_20"))
                record["rsi_14"] = safe_serialize(row.get("rsi_14"))
                record["macd_line"] = safe_serialize(row.get("macd_line"))
                record["macd_signal"] = safe_serialize(row.get("macd_signal"))
                record["macd_histogram"] = safe_serialize(row.get("macd_histogram"))
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

    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all validated datasets currently persisted in data/processed."""
        datasets = []
        for csv_path in sorted(self.processed_dir.glob("*_processed.csv")):
            stem = csv_path.stem.replace("_processed", "")
            parts = stem.split("_")
            interval = parts[-1] if len(parts) > 1 and parts[-1] in ("1d", "1h", "15m", "5m", "1m") else "1d"
            symbol_part = "_".join(parts[:-1]) if len(parts) > 1 and parts[-1] in ("1d", "1h", "15m", "5m", "1m") else stem
            symbol = symbol_part.upper().replace("_", ".")

            meta_file = PROJECT_ROOT / "data" / "metadata" / f"{csv_path.stem.replace('_processed', '')}_provenance.json"
            checksum = None
            if csv_path.exists():
                import hashlib
                sha = hashlib.sha256()
                with open(csv_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha.update(chunk)
                checksum = sha.hexdigest()

            summary = self._inspect_processed_data_summary(symbol) or {}
            datasets.append({
                "symbol": symbol,
                "interval": interval,
                "file_name": csv_path.name,
                "file_size_bytes": csv_path.stat().st_size,
                "checksum": checksum,
                "total_bars": summary.get("total_bars", 0),
                "start_date": summary.get("start_date"),
                "end_date": summary.get("end_date"),
                "has_provenance": meta_file.exists(),
            })
        return datasets

    def get_dataset_metadata(self, symbol: str, interval: str = "1d") -> Optional[Dict[str, Any]]:
        """Retrieve full provenance and validation metadata for a stored dataset."""
        clean_sym = validate_symbol(symbol)
        safe_sym = sanitize_filename_component(clean_sym)
        safe_int = sanitize_filename_component(interval)

        data_path = self.pipeline.get_canonical_processed_path(clean_sym, interval)
        meta_path = PROJECT_ROOT / "data" / "metadata" / f"{safe_sym}_{safe_int}_provenance.json"
        report_path = self.processed_dir / f"{safe_sym}_{safe_int}_processed.json"

        if not data_path.exists():
            return None

        # Checksum
        import hashlib
        sha = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        checksum = sha.hexdigest()

        provenance_dict = {}
        if meta_path.exists():
            import json
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    provenance_dict = json.load(f)
            except Exception:
                pass

        summary = self._inspect_processed_data_summary(clean_sym) or {}

        return {
            "symbol": clean_sym,
            "interval": interval,
            "file_name": data_path.name,
            "file_path": str(data_path.resolve()),
            "file_size_bytes": data_path.stat().st_size,
            "checksum": checksum,
            "total_bars": summary.get("total_bars", 0),
            "start_date": summary.get("start_date"),
            "end_date": summary.get("end_date"),
            "provenance": provenance_dict,
            "has_validation_report": report_path.exists(),
        }

    def validate_stored_dataset(self, symbol: str, interval: str = "1d") -> Dict[str, Any]:
        """Validate an on-disk dataset, verifying row consistency, gaps, and monotonicity."""
        from ..data.validator import validate_ohlcv_dataframe

        clean_sym = validate_symbol(symbol)
        data_path = self.pipeline.get_canonical_processed_path(clean_sym, interval)
        if not data_path.exists():
            return {
                "symbol": clean_sym,
                "status": "NOT_FOUND",
                "valid": False,
                "message": f"Processed dataset not found for '{clean_sym}'.",
            }

        try:
            df = pd.read_csv(data_path)
            clean_df, report = validate_ohlcv_dataframe(
                df, symbol=clean_sym, source="stored_verification", interval=interval
            )
            return {
                "symbol": clean_sym,
                "status": "VALID" if not report.has_rejected_rows else "WARNINGS",
                "valid": True,
                "total_rows": report.valid_row_count,
                "rejected_rows": report.rejected_row_count,
                "gaps_count": len(report.detected_gaps),
                "start_timestamp": report.start_timestamp,
                "end_timestamp": report.end_timestamp,
            }
        except Exception as exc:
            return {
                "symbol": clean_sym,
                "status": "CORRUPTED",
                "valid": False,
                "error": str(exc),
            }

    def refresh_symbol_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run incremental or force refresh for a symbol with failure isolation."""
        from ..data.market_config import MarketDataConfig

        clean_sym = validate_symbol(symbol)
        config_path = PROJECT_ROOT / "config" / "market_data_nse.json"
        config = MarketDataConfig.load_from_file(config_path) if config_path.exists() else None

        start = start_date or (config.date_range.start_date if config else "2016-01-01")
        end = end_date or (config.date_range.end_date if config else "2025-12-31")
        auto_adjust = config.price_adjustment.auto_adjust if config else True

        try:
            dest_path, report, prov = self.pipeline.download_and_process_symbol(
                symbol=clean_sym,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                refresh=True,
                save_report=True,
            )
            return {
                "success": True,
                "symbol": clean_sym,
                "status": prov.download_status,
                "validation_status": prov.validation_status,
                "valid_rows": report.valid_row_count,
                "start_timestamp": report.start_timestamp,
                "end_timestamp": report.end_timestamp,
                "file_path": str(dest_path.resolve()),
            }
        except Exception as exc:
            return {
                "success": False,
                "symbol": clean_sym,
                "error": str(exc),
            }

    def refresh_all_configured(self, interval: str = "1d") -> Dict[str, Any]:
        """Refresh all configured symbols defined in market_data_nse.json."""
        from ..data.market_config import MarketDataConfig

        config_path = PROJECT_ROOT / "config" / "market_data_nse.json"
        if not config_path.exists():
            return {"success": False, "error": "Configuration file not found."}

        config = MarketDataConfig.load_from_file(config_path)
        results = {}
        for sym in config.symbols:
            res = self.refresh_symbol_data(
                symbol=sym,
                interval=interval or config.frequency,
                start_date=config.date_range.start_date,
                end_date=config.date_range.end_date,
            )
            results[sym] = res

        successful = sum(1 for r in results.values() if r.get("success"))
        return {
            "success": successful == len(config.symbols),
            "total_configured": len(config.symbols),
            "successful_count": successful,
            "results": results,
        }

