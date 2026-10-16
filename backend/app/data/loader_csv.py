"""TradeMind CSV Data Ingestion Module.

Loads and normalizes raw OHLCV CSV files without modifying the original source files.
"""
import io
import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .exceptions import (
    DataFileNotFoundError,
    DataIngestionError,
    EmptyDataError,
    SchemaValidationError,
)
from .schema import REQUIRED_COLUMNS, normalize_column_name


class CSVDataLoader:
    """Reusable loader for raw historical OHLCV CSV datasets."""

    @staticmethod
    def load(
        source: Union[str, Path, io.StringIO, io.BytesIO],
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load an OHLCV CSV file into a pandas DataFrame with normalized columns.
        
        Args:
            source: File path (str/Path) or file-like buffer (StringIO/BytesIO).
            symbol: Optional symbol override if not inferred from filename.
            
        Returns:
            pd.DataFrame with lowercase normalized column names.
            
        Raises:
            DataFileNotFoundError: If a file path is specified but does not exist.
            EmptyDataError: If the file or buffer is completely empty or header-only.
            DataIngestionError: If pandas fails to parse the CSV format.
        """
        # 1. Path existence and size checks
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                raise DataFileNotFoundError(f"Data file not found at path: {path_obj.resolve()}")
            if path_obj.stat().st_size == 0:
                raise EmptyDataError(f"Data file is empty (0 bytes): {path_obj.name}")

        # 2. Read CSV via pandas
        try:
            df = pd.read_csv(source)
        except pd.errors.EmptyDataError:
            raise EmptyDataError("CSV dataset is empty or contains no readable records.")
        except Exception as exc:
            raise DataIngestionError(f"Failed to read CSV dataset: {str(exc)}") from exc

        # 3. Check for empty dataframe
        if df.empty:
            raise EmptyDataError("CSV dataset contains headers but 0 data rows.")

        # 4. Normalize column names (strip whitespace and map aliases)
        rename_map = {}
        for col in df.columns:
            normalized = normalize_column_name(str(col))
            rename_map[col] = normalized

        df = df.rename(columns=rename_map)

        # 5. Check if required columns are present after renaming
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise SchemaValidationError(
                f"CSV dataset is missing required OHLCV column(s): {missing_required}. "
                f"Found columns: {list(df.columns)}"
            )

        return df


def load_ohlcv_csv(
    source: Union[str, Path, io.StringIO, io.BytesIO],
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function wrapping CSVDataLoader.load."""
    return CSVDataLoader.load(source, symbol=symbol)
