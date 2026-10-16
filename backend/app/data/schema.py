"""TradeMind Canonical OHLCV Market Data Schema.

Schema definition, column constants, alias mappings, and timezone convention.

Timezone Convention:
    All timestamps in TradeMind are normalized to UTC (Coordinated Universal Time).
    - If incoming timestamps have timezone offsets (e.g. -05:00), they are converted to UTC.
    - If incoming timestamps are naive, they are localized to UTC.
    - Processed output files store ISO 8601 UTC formatted strings (e.g. '2024-01-02T00:00:00Z').
"""
from typing import Dict, List, Set

# Canonical Column Names
COL_TIMESTAMP = "timestamp"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"

# Metadata Column Names
COL_SYMBOL = "symbol"
COL_SOURCE = "source"

# Canonical Required OHLCV Columns
REQUIRED_COLUMNS: List[str] = [
    COL_TIMESTAMP,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
]

# Canonical Schema Types
NUMERIC_COLUMNS: List[str] = [
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
]

PRICE_COLUMNS: List[str] = [
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
]

# Common Column Alias Mappings (normalized to lower case for case-insensitive lookup)
COLUMN_ALIASES: Dict[str, str] = {
    # Timestamps
    "date": COL_TIMESTAMP,
    "datetime": COL_TIMESTAMP,
    "timestamp": COL_TIMESTAMP,
    "time": COL_TIMESTAMP,
    "dt": COL_TIMESTAMP,
    
    # Open
    "open": COL_OPEN,
    "open_price": COL_OPEN,
    
    # High
    "high": COL_HIGH,
    "high_price": COL_HIGH,
    
    # Low
    "low": COL_LOW,
    "low_price": COL_LOW,
    
    # Close
    "close": COL_CLOSE,
    "close_price": COL_CLOSE,
    "adj close": COL_CLOSE,
    "adj_close": COL_CLOSE,
    "adjusted_close": COL_CLOSE,
    
    # Volume
    "volume": COL_VOLUME,
    "vol": COL_VOLUME,
    "shares": COL_VOLUME,
    "total_volume": COL_VOLUME,
}


def normalize_column_name(col: str) -> str:
    """Map a column header string to its canonical OHLCV name if an alias exists.
    
    Args:
        col: Raw column header name.
        
    Returns:
        Canonical column name if recognized, otherwise trimmed lower-case original.
    """
    cleaned = col.strip().lower()
    return COLUMN_ALIASES.get(cleaned, cleaned)
