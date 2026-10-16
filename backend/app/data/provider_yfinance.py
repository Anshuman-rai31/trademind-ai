"""TradeMind Public Research Data Provider (yfinance integration).

Downloads and normalizes historical market data from Yahoo Finance.
Handles MultiIndex column structures, rate-limit retries, configurable corporate
action adjustments, and guarantees clean offline degradation.
"""
import time
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from .exceptions import (
    EmptyDataError,
    InvalidSymbolError,
    ProviderUnavailableError,
)
from .schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TIMESTAMP,
    COL_VOLUME,
    REQUIRED_COLUMNS,
)
from .validator import validate_symbol


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns if present.
    
    yfinance sometimes returns MultiIndex columns like:
    [('Close', 'RELIANCE.NS'), ('High', 'RELIANCE.NS'), ...]
    This normalizes them to single-level strings: ['Close', 'High', ...]
    """
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            # col can be a tuple like ('Close', 'RELIANCE.NS') or ('Date', '')
            if isinstance(col, tuple):
                # Pick the first non-empty element that is not the ticker if possible
                first = str(col[0]).strip()
                flat_cols.append(first)
            else:
                flat_cols.append(str(col))
        df = df.copy()
        df.columns = flat_cols
    return df


class YFinanceProvider:
    """Public research data provider fetching OHLCV data via yfinance with retries."""

    def __init__(self, timeout_seconds: int = 30, max_retries: int = 3, retry_backoff_factor: float = 1.5):
        """Initialize provider and verify yfinance library availability."""
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            self._yf = None

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor

    @property
    def is_available(self) -> bool:
        """Returns True if the yfinance dependency is installed and importable."""
        return self._yf is not None

    def fetch(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        save_raw_path: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for an asset symbol with retry resilience.
        
        Args:
            symbol: Ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'AAPL').
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD). Note: yfinance end date is exclusive.
            interval: Bar frequency ('1m', '5m', '15m', '1h', '1d').
            auto_adjust: If True, adjust OHLC prices backward for splits and dividends.
            save_raw_path: Optional path to save untouched raw response CSV before cleaning.
            
        Returns:
            pd.DataFrame with canonical OHLCV columns.
            
        Raises:
            ProviderUnavailableError: If yfinance is missing, rate limits hit, or network fails.
            InvalidSymbolError: If symbol format is invalid.
            EmptyDataError: If yfinance returns 0 records.
        """
        if not self.is_available:
            raise ProviderUnavailableError(
                "yfinance library is not installed or unavailable. "
                "Install via 'pip install yfinance' or use CSV datasets."
            )

        clean_symbol = validate_symbol(symbol)
        raw_df = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                ticker = self._yf.Ticker(clean_symbol)
                raw_df = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    timeout=self.timeout_seconds,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    raise ProviderUnavailableError(
                        f"Failed to fetch market data for '{clean_symbol}' from yfinance "
                        f"after {self.max_retries} attempts: {str(last_error)}"
                    ) from exc

        if raw_df is None or raw_df.empty:
            raise EmptyDataError(
                f"yfinance returned 0 records for symbol '{clean_symbol}'. "
                f"Verify the symbol exists on Yahoo Finance and dates ({start} to {end}) are valid trading sessions."
            )

        # Handle multi-level columns if present
        raw_df = flatten_yfinance_columns(raw_df)

        # Optionally save the raw provider dataframe untouched to raw_dir
        if save_raw_path:
            raw_path = Path(save_raw_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(raw_path)

        # Reset index to convert Datetime/Date index to a column
        df = raw_df.reset_index()
        df = flatten_yfinance_columns(df)

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            lower_col = str(col).lower()
            if lower_col in ["date", "datetime", "timestamp", "index"]:
                rename_map[col] = COL_TIMESTAMP
            elif lower_col == "open":
                rename_map[col] = COL_OPEN
            elif lower_col == "high":
                rename_map[col] = COL_HIGH
            elif lower_col == "low":
                rename_map[col] = COL_LOW
            elif lower_col in ["close", "adj close", "adj_close"]:
                rename_map[col] = COL_CLOSE
            elif lower_col in ["volume", "vol"]:
                rename_map[col] = COL_VOLUME

        df = df.rename(columns=rename_map)

        # Validate that essential columns are present
        available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        if len(available_cols) < len(REQUIRED_COLUMNS):
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            raise EmptyDataError(
                f"yfinance response for '{clean_symbol}' missing expected columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        return df[REQUIRED_COLUMNS].copy()


def fetch_ohlcv_provider(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Convenience function wrapping YFinanceProvider.fetch."""
    provider = YFinanceProvider()
    return provider.fetch(symbol, start=start, end=end, interval=interval, auto_adjust=auto_adjust)
