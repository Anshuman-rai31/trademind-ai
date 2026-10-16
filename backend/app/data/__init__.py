"""TradeMind Historical Market Data Pipeline Package (Phase 2 & NSE Config)."""

from .exceptions import (
    DataFileNotFoundError,
    DataIngestionError,
    DataValidationError,
    EmptyDataError,
    GapDetectionError,
    InsufficientDataError,
    InvalidSymbolError,
    ProviderUnavailableError,
    SchemaValidationError,
)
from .loader_csv import CSVDataLoader, load_ohlcv_csv
from .market_config import ConfigValidationError, MarketDataConfig
from .models import RejectedRow, TimeSeriesGap, ValidationReport
from .pipeline import MarketDataPipeline
from .provenance import ProvenanceMetadata
from .provider_yfinance import YFinanceProvider, fetch_ohlcv_provider, flatten_yfinance_columns
from .schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_SOURCE,
    COL_SYMBOL,
    COL_TIMESTAMP,
    COL_VOLUME,
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
    normalize_column_name,
)
from .split import ChronologicalSplits, SplitMetadata, chronological_split
from .validator import (
    detect_series_gaps,
    parse_and_normalize_timestamps,
    validate_ohlcv_dataframe,
    validate_symbol,
)

__all__ = [
    # Pipeline Orchestrator & Configuration
    "MarketDataPipeline",
    "MarketDataConfig",
    "ConfigValidationError",
    "ProvenanceMetadata",
    
    # Chronological Splitting (Phase 3 readiness)
    "chronological_split",
    "ChronologicalSplits",
    "SplitMetadata",
    
    # Loaders & Providers
    "CSVDataLoader",
    "load_ohlcv_csv",
    "YFinanceProvider",
    "fetch_ohlcv_provider",
    "flatten_yfinance_columns",
    
    # Validators
    "validate_ohlcv_dataframe",
    "validate_symbol",
    "detect_series_gaps",
    "parse_and_normalize_timestamps",
    
    # Models
    "ValidationReport",
    "TimeSeriesGap",
    "RejectedRow",
    
    # Schema Constants
    "COL_TIMESTAMP",
    "COL_OPEN",
    "COL_HIGH",
    "COL_LOW",
    "COL_CLOSE",
    "COL_VOLUME",
    "COL_SYMBOL",
    "COL_SOURCE",
    "REQUIRED_COLUMNS",
    "PRICE_COLUMNS",
    "normalize_column_name",
    
    # Exceptions
    "DataIngestionError",
    "DataFileNotFoundError",
    "EmptyDataError",
    "SchemaValidationError",
    "DataValidationError",
    "InsufficientDataError",
    "InvalidSymbolError",
    "ProviderUnavailableError",
    "GapDetectionError",
]
