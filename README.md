# TradeMind: AI-Powered Quantitative Trading Research & Simulation Platform

TradeMind is an institutional-grade quantitative trading research, feature engineering, and reinforcement learning simulation environment configured for Indian equities listed on the **National Stock Exchange (NSE)**.

---

## 📌 Platform Architecture & Roadmap

- **Phase 1: Project Setup & Baseline** (Completed)
- **Phase 2: Historical Market Data Pipeline & NSE Configuration** (Completed)
- **Phase 3: Feature Engineering & Preprocessing** (Next Phase)
- **Phase 4: Gymnasium Simulation Environment & PPO RL Engine** (Pending)

> **Simulation Boundary**: TradeMind is designed exclusively for offline historical research, feature engineering, and RL simulation. It does **not** connect to live brokerage APIs or execute real-money orders.

---

## 📁 Repository Directory Structure

```text
TradeMind/
├── .gitignore                      # Git exclusion rules (venv, checkpoints, databases, data)
├── requirements.txt                # Pinned dependencies with PyTorch index configuration
├── run.py                          # Development Flask server launcher
├── process_dataset.py              # CLI entry point for data ingestion, validation & splitting
├── README.md                       # Comprehensive platform documentation
├── config/
│   └── market_data_nse.json        # Central market data configuration for NSE equities
├── backend/
│   └── app/
│       ├── __init__.py             # Flask application factory (create_app)
│       ├── config.py               # Config classes (Development, Testing, Production)
│       ├── data/                   # Market Data Pipeline Package
│       │   ├── __init__.py         # Package exports
│       │   ├── cli.py              # Command-line interface logic
│       │   ├── exceptions.py       # Actionable domain exceptions
│       │   ├── loader_csv.py       # Raw OHLCV CSV loader
│       │   ├── market_config.py    # Central config loader and validator
│       │   ├── models.py           # ValidationReport, TimeSeriesGap, RejectedRow
│       │   ├── pipeline.py         # Pipeline coordinator (MarketDataPipeline)
│       │   ├── provenance.py       # ProvenanceMetadata tracking and persistence
│       │   ├── provider_yfinance.py# Public research data provider (yfinance)
│       │   ├── schema.py           # Canonical schema, aliases, timezone constants
│       │   ├── split.py            # Chronological 70/20/10 dataset splitting utility
│       │   └── validator.py        # Strict validation & gap detection engine
│       └── routes/
│           ├── __init__.py         # Blueprint exports
│           ├── main.py             # Landing page route (/)
│           └── health.py           # Health-check route (/health, /api/health)
├── data/
│   ├── raw/                        # Untouched raw downloaded & uploaded CSVs (.gitkeep)
│   │   └── sample_aapl_daily.csv   # Deterministic sample dataset
│   ├── processed/                  # Normalized, validated datasets (.gitkeep)
│   └── metadata/                   # Provenance JSON audit trails (.gitkeep)
├── instance/                       # Local SQLite storage and runtime instance secrets (.gitkeep)
├── models/                         # ML weights, PyTorch checkpoints, RL policies (.gitkeep)
├── notebooks/                      # Jupyter research & exploratory analysis notebooks (.gitkeep)
├── static/
│   ├── css/
│   │   └── style.css               # Modern dark-theme stylesheet
│   └── js/
│       └── main.js                 # Client-side health diagnostics script
├── templates/
│   ├── base.html                   # Base HTML5 layout
│   └── index.html                  # TradeMind landing page
└── tests/
    ├── __init__.py                 # Test package initialization
    ├── conftest.py                 # Pytest fixtures (client, app)
    ├── test_data_pipeline.py       # Ingestion and validation unit tests
    ├── test_health.py              # Flask route and asset tests
    └── test_nse_pipeline.py        # Central config, NSE download, refresh, and split tests
```

---

## ⚙️ Central Configuration (`config/market_data_nse.json`)

The platform is driven by a central, validated configuration file:

```json
{
  "market": "NSE",
  "frequency": "1d",
  "data_source": "yfinance",
  "symbols": [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS"
  ],
  "date_range": {
    "start_date": "2016-01-01",
    "end_date": "2025-12-31"
  },
  "price_adjustment": {
    "auto_adjust": true,
    "policy_description": "Uses yfinance auto_adjust=True to backward-adjust OHLC bars for corporate stock splits and dividends, preserving relative percentage returns."
  },
  "storage_paths": {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "metadata_dir": "data/metadata"
  },
  "split_proportions": {
    "train": 0.70,
    "validation": 0.20,
    "test": 0.10
  },
  "download_settings": {
    "timeout_seconds": 30,
    "max_retries": 3,
    "retry_backoff_factor": 1.5
  },
  "future_rl_algorithm": "PPO",
  "notes": "Historical simulation only."
}
```

### Configuration Validation Rules:
- **`market`**: Non-empty identifier (e.g. `NSE`).
- **`frequency`**: Must be one of `['1m', '5m', '15m', '30m', '1h', '1d', '1w']`.
- **`symbols`**: Validated against alphanumeric syntax and exchange suffixes (`.NS`).
- **`date_range`**: Valid `YYYY-MM-DD` strings with `start_date < end_date`. (Note: `yfinance` treats end date as exclusive).
- **`split_proportions`**: Floats in $(0, 1)$ strictly summing to $1.0$ (initial defaults: 70% Train, 20% Validation, 10% Test).
- **`download_settings`**: Positive timeouts and finite retry bounds avoiding infinite retry loops.

---

## 📊 Canonical Data Schema & Timezone Convention

### Processed Dataset Schema
Every processed dataset provides a standardized, tabular format suitable for feature engineering:

| Column Name | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `ISO 8601 String` / `datetime64[ns, UTC]` | UTC normalized, monotonic ascending | Bar session timestamp |
| `open` | `Float64` | `open > 0` | Opening price |
| `high` | `Float64` | `high > 0`, `high >= low`, `high >= open`, `high >= close` | Period high price |
| `low` | `Float64` | `low > 0`, `low <= high`, `low <= open`, `low <= close` | Period low price |
| `close` | `Float64` | `close > 0` | Closing price |
| `volume` | `Float64` / `Int64` | `volume >= 0` | Traded share volume |
| `symbol` | `String` (Metadata) | Uppercase ticker (e.g. `RELIANCE.NS`) | Asset identifier |
| `source` | `String` (Metadata) | e.g. `yfinance` or `csv:filename` | Data origin |

### Timezone Convention
- All timestamps are normalized to **UTC (Coordinated Universal Time)**.
- IST (Indian Standard Time, UTC+05:30) trading sessions are converted to standard UTC ISO 8601 strings (`...Z`).
- Regional daylight saving shifts are non-applicable for Indian equities, but UTC avoids cross-border and leap-second ambiguities.

---

## 📈 Corporate Actions & Price Adjustment Policy

- **Adjustment Basis**: When `auto_adjust=True`, historical OHLC prices are adjusted backward for stock splits, bonus issues, and dividends. This prevents artificial price cliffs (e.g., a 1:1 bonus issue dropping nominal price by 50%) that would corrupt return calculations and gradient steps in reinforcement learning.
- **Consistency**: Open, High, Low, and Close are adjusted using the same corporate action factor. We never mix adjusted close with unadjusted open/high/low.
- **Raw Data Preservation**: Unaltered provider outputs (including unadjusted historical bars where provided) are written to `data/raw/` before any transformation occurs.
- **Limitations**: Yahoo Finance adjustments are backward-looking approximations and may differ slightly from official NSE corporate action circulars. Dividend cash flow reinvestment is approximated via ratio adjustments rather than discrete cash ledger tracking.

---

## 🔄 Refresh & Duplicate Protection Mechanism

When running an explicit update or download:
1. **Cache Inspection**: If `refresh=False`, the pipeline checks for existing validated processed data and metadata covering the requested date range, avoiding redundant downloads.
2. **Untouched Raw Save**: Influx raw data is saved with a timestamped filename in `data/raw/` (`{symbol}_{freq}_raw_{timestamp}.csv`).
3. **Deterministic Merge**: New records are merged with existing records using `timestamp` as the primary key. Overlapping sessions are resolved by retaining the freshest download.
4. **Zero Row Multiplication**: Safe to repeat repeatedly without row inflation.
5. **Atomic Validation**: The entire merged series is re-validated. The processed CSV and provenance metadata are updated only if validation passes 100%.
6. **Failure Isolation**: If a download or network request fails, previously validated data in `data/processed/` is left completely intact.

---

## 🚀 Windows PowerShell Commands

### 1. Validate Central Configuration
```powershell
.\.venv\Scripts\python.exe process_dataset.py --config config/market_data_nse.json --validate-config
```

### 2. Download and Validate a Single NSE Symbol
```powershell
.\.venv\Scripts\python.exe process_dataset.py --config config/market_data_nse.json --symbol RELIANCE.NS --start 2016-01-01 --end 2025-12-31
```

### 3. Force-Refresh Existing Historical Data
```powershell
.\.venv\Scripts\python.exe process_dataset.py --config config/market_data_nse.json --symbol TCS.NS --refresh
```

### 4. Batch Download All Configured NSE Symbols
```powershell
.\.venv\Scripts\python.exe process_dataset.py --config config/market_data_nse.json --download-all
```

### 5. Ingest Local OHLCV CSV Dataset
```powershell
.\.venv\Scripts\python.exe process_dataset.py --file data/raw/sample_aapl_daily.csv --symbol AAPL --interval 1d
```

### 6. Run Chronological 70/20/10 Dataset Split
```powershell
.\.venv\Scripts\python.exe process_dataset.py --file data/processed/reliance_ns_1d_processed.csv --split
```

---

## 🧪 Automated Testing

The automated test suite runs 100% offline without live internet dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Coverage (34 Tests):
- Configuration syntax, date range, frequency, and split proportion validation.
- Mocked `yfinance` download handling and network retry/timeout resilience.
- Handling MultiIndex column structures from `yfinance`.
- Price positivity (`> 0`), non-negative volume (`>= 0`), and relational bounds (`high >= low`, `high >= open`, `low <= close`).
- Missing value reporting and time-series gap detection without synthetic filling.
- Raw file immutability and provenance metadata recording.
- Safe refresh, deduplication, and failure isolation.
- Chronological splitting, strict temporal ordering, and anti-leakage embargo bars.
- Flask application factory and health check endpoints (`/health` and `/api/health`).

---

## 🔮 Phase 3 Readiness: Feature Engineering Interface

The validated dataset is ready for Phase 3 feature engineering.

### Expected Downstream Calculations:
- **Returns**: Simple returns ($\frac{P_t - P_{t-1}}{P_{t-1}}$) and log returns ($\ln(P_t / P_{t-1})$).
- **Moving Averages**: Fast/slow SMAs and EMAs (e.g., 10, 20, 50-period).
- **Volatility**: Rolling sample standard deviation of log returns.
- **Momentum & Trend**: RSI (Relative Strength Index, 14-period), MACD, and Bollinger Bands.
- **Volume Indicators**: On-Balance Volume (OBV), Volume Weighted Average Price (VWAP).

### Mandatory Phase 3 Constraints:
1. **Zero Future Leakage**: All rolling calculations must compute features using only current and historical observations ($t \le T$).
2. **Pre-Split Fitting**: Normalization scalers (e.g. `StandardScaler`, `MinMaxScaler`) must be fit **exclusively on the training partition** and applied to validation/test sets without refitting.
3. **Embargo Buffers**: When generating indicators with lookback window $W$, the chronological split utility's `embargo_bars=W` parameter should be used to purge $W$ bars prior to partition transitions, guaranteeing zero boundary bleed.
4. **Final Test Isolation**: The final 10% test period must remain strictly untouched during hyperparameter tuning and model selection.

---

## ⚠️ Data Limitations & Licensing Considerations

1. **Development Source**: Yahoo Finance (`yfinance`) is used as an initial development and research data provider. It is **not** an official NSE data feed.
2. **Official NSE Data Integration Point**: For production backtesting or institutional trade simulation, users can drop official Bhavcopy CSV files or licensed vendor feeds into `data/raw/` and process them through the same validation pipeline via `process_dataset.py --file <path>`.
3. **Trading Holidays & Gaps**: The pipeline detects gaps between consecutive trading sessions (e.g. weekends, Diwali, Republic Day) and logs them in [`ValidationReport.detected_gaps`](file:///c:/TradeMind/backend/app/data/models.py#L10-L24). It does **not** invent synthetic bars for non-trading sessions.
4. **No Live Execution**: TradeMind does not store brokerage API keys or route orders to exchanges.
