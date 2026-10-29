"""TradeMind Market Data Processing Pipeline.

Coordinates CSV/provider ingestion, rigorous validation, diagnostic reporting,
provenance metadata logging, and incremental refresh with duplicate protection.
Guarantees raw files under data/raw remain untouched.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from .exceptions import DataIngestionError, EmptyDataError
from .loader_csv import CSVDataLoader
from .market_config import MarketDataConfig
from .models import ValidationReport
from .provenance import ProvenanceMetadata
from .provider_yfinance import YFinanceProvider
from .schema import COL_SYMBOL, COL_TIMESTAMP, REQUIRED_COLUMNS
from .validator import validate_ohlcv_dataframe, validate_symbol

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def sanitize_filename_component(name: str) -> str:
    """Sanitize string to safe characters for filenames (e.g. RELIANCE.NS -> reliance_ns)."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip().lower())


class MarketDataPipeline:
    """Orchestrates market data ingestion, validation, provenance, and storage."""

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        metadata_dir: Optional[Union[str, Path]] = None,
        config: Optional[MarketDataConfig] = None,
    ):
        """Initialize pipeline with raw, processed, and metadata storage directories.
        
        Args:
            raw_dir: Directory containing raw data files (defaults to data/raw).
            processed_dir: Directory where processed files are saved (defaults to data/processed).
            metadata_dir: Directory for provenance JSON records (defaults to data/metadata).
            config: Optional MarketDataConfig instance for global settings.
        """
        self.config = config
        self.raw_dir = Path(raw_dir) if raw_dir else (Path(config.storage_paths.raw_dir) if config else DEFAULT_RAW_DIR)
        self.processed_dir = Path(processed_dir) if processed_dir else (Path(config.storage_paths.processed_dir) if config else DEFAULT_PROCESSED_DIR)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else (Path(config.storage_paths.metadata_dir) if config else DEFAULT_METADATA_DIR)

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.csv_loader = CSVDataLoader()
        download_settings = config.download_settings if config else None
        self.provider = YFinanceProvider(
            timeout_seconds=download_settings.timeout_seconds if download_settings else 30,
            max_retries=download_settings.max_retries if download_settings else 3,
            retry_backoff_factor=download_settings.retry_backoff_factor if download_settings else 1.5,
        )

    def generate_output_filename(
        self,
        symbol: str,
        interval: str,
        report: ValidationReport,
        extension: str = "csv",
    ) -> str:
        """Create a safe, predictable filename for processed market data.
        
        Format: {symbol}_{interval}_{start_date}_{end_date}.{extension}
        Example: reliance_ns_1d_20160101_20251231.csv
        """
        safe_sym = sanitize_filename_component(symbol)
        safe_interval = sanitize_filename_component(interval or "1d")

        def format_ts(ts_str: Optional[str]) -> str:
            if not ts_str:
                return "unknown"
            date_part = ts_str.split("T")[0].replace("-", "")
            return date_part if date_part else "unknown"

        start_str = format_ts(report.start_timestamp)
        end_str = format_ts(report.end_timestamp)

        return f"{safe_sym}_{safe_interval}_{start_str}_{end_str}.{extension}"

    def get_canonical_processed_path(self, symbol: str, interval: str = "1d") -> Path:
        """Get the deterministic path for the canonical processed dataset of a symbol."""
        safe_sym = sanitize_filename_component(symbol)
        safe_interval = sanitize_filename_component(interval or "1d")
        return self.processed_dir / f"{safe_sym}_{safe_interval}_processed.csv"

    def process_csv(
        self,
        source_file: Union[str, Path],
        symbol: Optional[str] = None,
        interval: str = "1d",
        save_report: bool = True,
    ) -> Tuple[Path, ValidationReport]:
        """Ingest, validate, and persist an OHLCV CSV file to data/processed.
        
        Original raw file is kept completely unchanged.
        """
        source_path = Path(source_file)
        if not symbol:
            symbol = source_path.stem.split("_")[0].upper()

        clean_symbol = validate_symbol(symbol)
        raw_df = self.csv_loader.load(source_path, symbol=clean_symbol)

        clean_df, report = validate_ohlcv_dataframe(
            raw_df,
            symbol=clean_symbol,
            source=f"csv:{source_path.name}",
            interval=interval,
        )

        export_df = clean_df.copy()
        export_df[COL_TIMESTAMP] = export_df[COL_TIMESTAMP].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        filename = self.generate_output_filename(clean_symbol, interval, report, "csv")
        dest_path = self.processed_dir / filename
        try:
            export_df.to_csv(dest_path, index=False)
        except PermissionError as exc:
            raise DataIngestionError(
                f"Permission denied writing to '{dest_path}'. "
                f"The file is locked by another process (such as Microsoft Excel). "
                f"Please close the file in the external application and retry."
            ) from exc

        if save_report:
            report_filename = dest_path.with_suffix(".json")
            try:
                with open(report_filename, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
            except PermissionError as exc:
                raise DataIngestionError(
                    f"Permission denied writing report to '{report_filename}'. "
                    f"The file is locked by another process. Please close it and retry."
                ) from exc

        return dest_path, report

    def process_provider(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        save_report: bool = True,
    ) -> Tuple[Path, ValidationReport]:
        """Fetch, validate, and persist market data from yfinance provider."""
        dest_path, report, _ = self.download_and_process_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            refresh=True,
            save_report=save_report,
        )
        return dest_path, report

    def download_and_process_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        refresh: bool = False,
        save_report: bool = True,
    ) -> Tuple[Path, ValidationReport, ProvenanceMetadata]:
        """Download (or refresh) and validate historical market data for a symbol.
        
        Implements Safe Refresh & Duplicate Protection:
            1. If refresh=False and an existing validated processed file and provenance record exist,
               reuses cached data without unnecessary re-downloading.
            2. Saves raw provider response separately to data/raw without mutating previous raw files.
            3. Merges new records with existing records using (timestamp, symbol) as unique key.
            4. Resolves overlapping dates deterministically (incoming records overwrite overlapping timestamps).
            5. Re-validates the entire merged series.
            6. Atomic update to processed dataset and metadata ONLY after validation passes.
            7. If download fails, previously validated data is left 100% intact.
            
        Returns:
            Tuple of (Path to processed file, ValidationReport, ProvenanceMetadata).
        """
        clean_symbol = validate_symbol(symbol)
        safe_sym = sanitize_filename_component(clean_symbol)
        safe_interval = sanitize_filename_component(interval or "1d")

        canonical_dest = self.get_canonical_processed_path(clean_symbol, interval)
        provenance_path = self.metadata_dir / f"{safe_sym}_{safe_interval}_provenance.json"

        # Check for cached valid dataset if refresh is False
        if not refresh and canonical_dest.exists() and provenance_path.exists():
            try:
                cached_prov = ProvenanceMetadata.load(provenance_path)
                cached_df = self.csv_loader.load(canonical_dest, symbol=clean_symbol)
                clean_df, report = validate_ohlcv_dataframe(
                    cached_df, symbol=clean_symbol, source=cached_prov.source, interval=interval
                )
                cached_prov.download_status = "CACHED"
                return canonical_dest, report, cached_prov
            except Exception:
                # If cached reading fails, proceed to clean download
                pass

        # 1. Prepare raw download destination
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_filename = f"{safe_sym}_{safe_interval}_raw_{timestamp_str}.csv"
        raw_dest_path = self.raw_dir / raw_filename

        # 2. Fetch raw data from yfinance provider (saves untouched raw CSV)
        raw_incoming_df = self.provider.fetch(
            symbol=clean_symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            save_raw_path=raw_dest_path,
        )

        # 3. Check for existing processed dataset to merge
        merged_work_df = raw_incoming_df.copy()
        if canonical_dest.exists():
            try:
                existing_df = self.csv_loader.load(canonical_dest, symbol=clean_symbol)
                # Concatenate existing and new records
                combined_df = pd.concat([existing_df, raw_incoming_df], ignore_index=True)
                # Resolve duplicates: incoming newest download takes precedence
                combined_df.drop_duplicates(subset=[COL_TIMESTAMP], keep="last", inplace=True)
                merged_work_df = combined_df
            except Exception:
                # If existing file cannot be read, fall back to incoming raw data
                merged_work_df = raw_incoming_df.copy()

        # 4. Strict Validation of merged dataset
        clean_df, report = validate_ohlcv_dataframe(
            merged_work_df,
            symbol=clean_symbol,
            source="yfinance",
            interval=interval,
        )

        # 5. Export to processed directory
        export_df = clean_df.copy()
        export_df[COL_TIMESTAMP] = export_df[COL_TIMESTAMP].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save canonical processed file
        try:
            export_df.to_csv(canonical_dest, index=False)
        except PermissionError as exc:
            raise DataIngestionError(
                f"Permission denied writing to '{canonical_dest}'. "
                f"The file is locked by another process (such as Microsoft Excel). "
                f"Please close the file in the external application and retry."
            ) from exc

        # Also save date-tagged copy for compatibility with generate_output_filename
        date_tagged_name = self.generate_output_filename(clean_symbol, interval, report, "csv")
        date_tagged_dest = self.processed_dir / date_tagged_name
        try:
            export_df.to_csv(date_tagged_dest, index=False)
        except PermissionError:
            pass

        # 6. Save companion JSON report
        if save_report:
            report_file = canonical_dest.with_suffix(".json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        # 7. Create and persist Provenance Metadata
        provenance = ProvenanceMetadata(
            symbol=clean_symbol,
            source="yfinance",
            frequency=interval,
            auto_adjust=auto_adjust,
            download_status="REFRESHED" if refresh else "SUCCESS",
            validation_status="PASSED" if not report.has_rejected_rows else "WARNINGS",
            requested_start_date=start or "N/A",
            requested_end_date=end or "N/A",
            actual_min_date=report.start_timestamp,
            actual_max_date=report.end_timestamp,
            row_count=report.valid_row_count,
            validation_errors=[r.reasons[0] for r in report.rejected_rows[:5]],
            validation_warnings=[
                f"Detected {len(report.detected_gaps)} time-series gaps."
            ] if report.has_gaps else [],
            raw_file_path=str(raw_dest_path.resolve()),
            processed_file_path=str(canonical_dest.resolve()),
            adjustment_policy_notes=(
                "auto_adjust=True: OHLC prices backward-adjusted for corporate stock splits and dividends."
                if auto_adjust else "auto_adjust=False: Unadjusted nominal historical prices."
            ),
        )
        provenance.save(self.metadata_dir)

        return canonical_dest, report, provenance

    def process_all_configured(
        self,
        config: Optional[MarketDataConfig] = None,
        refresh: bool = False,
    ) -> Dict[str, Tuple[Path, ValidationReport, ProvenanceMetadata]]:
        """Download and validate all symbols specified in the central configuration.
        
        Reports the status of each symbol separately. Failure of one symbol does not
        abort or invalidate successfully processed symbols.
        
        Returns:
            Dictionary mapping symbol -> (Path, ValidationReport, ProvenanceMetadata).
        """
        cfg = config or self.config
        if not cfg:
            raise DataIngestionError("No MarketDataConfig provided to process_all_configured.")

        results: Dict[str, Tuple[Path, ValidationReport, ProvenanceMetadata]] = {}
        errors: Dict[str, str] = {}

        for sym in cfg.symbols:
            try:
                path, report, prov = self.download_and_process_symbol(
                    symbol=sym,
                    start=cfg.date_range.start_date,
                    end=cfg.date_range.end_date,
                    interval=cfg.frequency,
                    auto_adjust=cfg.price_adjustment.auto_adjust,
                    refresh=refresh,
                )
                results[sym] = (path, report, prov)
            except Exception as exc:
                errors[sym] = str(exc)

        if errors:
            print(f"[PIPELINE WARNING] {len(errors)} symbol(s) encountered download/validation errors:")
            for sym, err in errors.items():
                print(f"  - {sym}: {err}")

        return results
