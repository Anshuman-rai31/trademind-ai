"""TradeMind Market Data Configuration Loader and Validator.

Loads and validates central market data configuration files (JSON).
Ensures parameters for symbols, date ranges, storage paths, and chronological
train/validation/test split ratios meet strict constraints before pipeline execution.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import DataIngestionError
from .validator import validate_symbol


class ConfigValidationError(DataIngestionError):
    """Raised when configuration parameters fail validation."""
    pass


@dataclass
class SplitProportions:
    train: float = 0.70
    validation: float = 0.20
    test: float = 0.10

    def validate(self) -> None:
        for name, val in [("train", self.train), ("validation", self.validation), ("test", self.test)]:
            if not isinstance(val, (int, float)) or val <= 0 or val >= 1.0:
                raise ConfigValidationError(
                    f"Split proportion '{name}' must be a float between 0 and 1, got: {val}"
                )
        total = round(self.train + self.validation + self.test, 6)
        if abs(total - 1.0) > 1e-5:
            raise ConfigValidationError(
                f"Split proportions (train={self.train}, val={self.validation}, test={self.test}) "
                f"must sum to 1.0, but sum to {total}"
            )


@dataclass
class DateRange:
    start_date: str = "2016-01-01"
    end_date: str = "2025-12-31"

    def validate(self) -> None:
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        for field_name, d_str in [("start_date", self.start_date), ("end_date", self.end_date)]:
            if not isinstance(d_str, str) or not re.match(date_pattern, d_str.strip()):
                raise ConfigValidationError(
                    f"'{field_name}' must be in YYYY-MM-DD format, got: {d_str}"
                )
            try:
                datetime.strptime(d_str.strip(), "%Y-%m-%d")
            except ValueError as exc:
                raise ConfigValidationError(f"Invalid date for '{field_name}': {d_str}") from exc

        d_start = datetime.strptime(self.start_date.strip(), "%Y-%m-%d")
        d_end = datetime.strptime(self.end_date.strip(), "%Y-%m-%d")
        if d_start >= d_end:
            raise ConfigValidationError(
                f"start_date ({self.start_date}) must be earlier than end_date ({self.end_date})"
            )


@dataclass
class StoragePaths:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"

    def validate(self) -> None:
        for name, p in [("raw_dir", self.raw_dir), ("processed_dir", self.processed_dir), ("metadata_dir", self.metadata_dir)]:
            if not p or not isinstance(p, str):
                raise ConfigValidationError(f"Storage path '{name}' must be a non-empty string.")


@dataclass
class DownloadSettings:
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.5

    def validate(self) -> None:
        if not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise ConfigValidationError(f"timeout_seconds must be a positive integer, got: {self.timeout_seconds}")
        if not isinstance(self.max_retries, int) or self.max_retries < 0 or self.max_retries > 10:
            raise ConfigValidationError(f"max_retries must be between 0 and 10, got: {self.max_retries}")
        if not isinstance(self.retry_backoff_factor, (int, float)) or self.retry_backoff_factor < 1.0:
            raise ConfigValidationError(f"retry_backoff_factor must be >= 1.0, got: {self.retry_backoff_factor}")


@dataclass
class PriceAdjustmentConfig:
    auto_adjust: bool = True
    policy_description: str = "Uses yfinance auto_adjust=True to backward-adjust OHLC bars for corporate actions."


@dataclass
class MarketDataConfig:
    """Central configuration for TradeMind market data pipeline."""
    market: str = "NSE"
    frequency: str = "1d"
    data_source: str = "yfinance"
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"
    ])
    date_range: DateRange = field(default_factory=DateRange)
    price_adjustment: PriceAdjustmentConfig = field(default_factory=PriceAdjustmentConfig)
    storage_paths: StoragePaths = field(default_factory=StoragePaths)
    split_proportions: SplitProportions = field(default_factory=SplitProportions)
    download_settings: DownloadSettings = field(default_factory=DownloadSettings)
    future_rl_algorithm: str = "PPO"
    notes: str = "Historical simulation only."

    def validate(self) -> None:
        """Validate all parameters in this configuration."""
        if not self.market or not isinstance(self.market, str):
            raise ConfigValidationError("Market must be specified as a non-empty string (e.g. 'NSE').")

        allowed_frequencies = {"1m", "2m", "5m", "15m", "30m", "1h", "1d", "1D", "1w", "1W"}
        if self.frequency not in allowed_frequencies:
            raise ConfigValidationError(
                f"Unsupported frequency '{self.frequency}'. Allowed: {sorted(allowed_frequencies)}"
            )

        if not self.symbols or not isinstance(self.symbols, list):
            raise ConfigValidationError("symbols must be a non-empty list of ticker symbols.")

        for sym in self.symbols:
            try:
                validate_symbol(sym)
            except Exception as exc:
                raise ConfigValidationError(f"Invalid symbol in configuration: {sym} ({str(exc)})") from exc

        self.date_range.validate()
        self.storage_paths.validate()
        self.split_proportions.validate()
        self.download_settings.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketDataConfig":
        """Build and validate a MarketDataConfig instance from a dictionary."""
        date_range_data = data.get("date_range", {})
        date_range = DateRange(
            start_date=date_range_data.get("start_date", "2016-01-01"),
            end_date=date_range_data.get("end_date", "2025-12-31"),
        )

        storage_data = data.get("storage_paths", {})
        storage_paths = StoragePaths(
            raw_dir=storage_data.get("raw_dir", "data/raw"),
            processed_dir=storage_data.get("processed_dir", "data/processed"),
            metadata_dir=storage_data.get("metadata_dir", "data/metadata"),
        )

        split_data = data.get("split_proportions", {})
        split_proportions = SplitProportions(
            train=split_data.get("train", 0.70),
            validation=split_data.get("validation", 0.20),
            test=split_data.get("test", 0.10),
        )

        download_data = data.get("download_settings", {})
        download_settings = DownloadSettings(
            timeout_seconds=download_data.get("timeout_seconds", 30),
            max_retries=download_data.get("max_retries", 3),
            retry_backoff_factor=download_data.get("retry_backoff_factor", 1.5),
        )

        adj_data = data.get("price_adjustment", {})
        price_adj = PriceAdjustmentConfig(
            auto_adjust=bool(adj_data.get("auto_adjust", True)),
            policy_description=adj_data.get("policy_description", ""),
        )

        config = cls(
            market=data.get("market", "NSE"),
            frequency=data.get("frequency", "1d"),
            data_source=data.get("data_source", "yfinance"),
            symbols=data.get("symbols", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]),
            date_range=date_range,
            price_adjustment=price_adj,
            storage_paths=storage_paths,
            split_proportions=split_proportions,
            download_settings=download_settings,
            future_rl_algorithm=data.get("future_rl_algorithm", "PPO"),
            notes=data.get("notes", "Historical simulation only."),
        )
        config.validate()
        return config

    @classmethod
    def load_from_file(cls, path: str | Path) -> "MarketDataConfig":
        """Load and validate configuration from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigValidationError(f"Configuration file not found: {file_path.resolve()}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Malformed JSON in configuration file '{file_path.name}': {str(exc)}") from exc
        except Exception as exc:
            raise ConfigValidationError(f"Error reading configuration file '{file_path.name}': {str(exc)}") from exc

        return cls.from_dict(data)
