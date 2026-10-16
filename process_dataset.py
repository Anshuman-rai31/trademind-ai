"""TradeMind Root Dataset Processing Script.

Convenience launcher for the Phase 2 Historical Market Data Pipeline.

Examples:
    python process_dataset.py --file data/raw/sample.csv --symbol AAPL --interval 1d
    python process_dataset.py --provider yfinance --symbol SPY --start 2024-01-01 --end 2024-03-01
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.data.cli import main

if __name__ == "__main__":
    sys.exit(main())
