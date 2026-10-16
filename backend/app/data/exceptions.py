"""TradeMind Market Data Pipeline Exceptions.

Hierarchy of domain-specific actionable exceptions for data ingestion,
schema validation, numeric verification, and provider communication.
"""


class DataIngestionError(Exception):
    """Base exception for all TradeMind data ingestion errors."""
    pass


class DataFileNotFoundError(DataIngestionError):
    """Raised when the specified source data file does not exist."""
    pass


class EmptyDataError(DataIngestionError):
    """Raised when the source dataset is completely empty or contains no records."""
    pass


class SchemaValidationError(DataIngestionError):
    """Raised when source data fails required schema constraints or missing columns."""
    pass


class DataValidationError(DataIngestionError):
    """Raised when data values fail integrity rules (e.g. non-numeric, all rows invalid)."""
    pass


class InsufficientDataError(DataIngestionError):
    """Raised when the valid record count is below the minimum required threshold."""
    pass


class InvalidSymbolError(DataIngestionError):
    """Raised when an asset symbol is syntactically invalid or unsafe for filenames."""
    pass


class ProviderUnavailableError(DataIngestionError):
    """Raised when an external research provider (e.g. yfinance) is uninstalled or unreachable."""
    pass


class GapDetectionError(DataIngestionError):
    """Raised when interval or frequency gap analysis fails."""
    pass
