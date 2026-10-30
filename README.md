# TradeMind: AI-Powered Quantitative Trading Research & Simulation Platform

TradeMind is an institutional-grade quantitative trading research, feature engineering, and reinforcement learning simulation environment configured for Indian equities listed on the **National Stock Exchange (NSE)**.

---

## 📌 Platform Architecture & Roadmap

- **Phase 1: Project Setup & Baseline** (Completed)
- **Phase 2: Historical Market Data Pipeline & NSE Configuration** (Completed)
- **Phase 3: Feature Engineering & Preprocessing** (Completed)
- **Phase 4: Baseline Backtesting & Portfolio Simulator** (Completed)
- **Phase 5: Gymnasium Trading Environment** (Completed)
- **Phase 6: PPO Reinforcement Learning Training Pipeline** (Completed)
- **Phase 7: Institutional Risk Management Engine** (Completed)
- **Phase 8: Unsupervised Market-Regime Detection** (Completed)
- **Phase 9: Flask REST API & SQLite Database Integration** (Completed)
- **Phase 9.5A: Historical Market Data Recovery & Validation** (Completed)
- **Phase 9.5B: Persistent Storage, Database Integration & End-to-End System Integration** (Completed)
- **Phase 10A: Interactive Market Visualization (Candlestick, Volume & Indicators)** (Completed)
- **Phase 11A–11G: Real-Time Market Data Provider, Ingestion, Candle Construction & Dashboard** (Completed)
- **Phase 12: Real-Time RL Policy Trading Signals & Risk Gating** (Completed)

> **Simulation Boundary**: TradeMind is designed exclusively for offline historical research, feature engineering, and RL simulation. It does **not** connect to live brokerage APIs for order routing or execute real-money orders. No profitability claims are made.

---

## 📁 Repository Directory Structure

```text
TradeMind/
├── .gitignore                      # Git exclusion rules (venv, checkpoints, databases, data)
├── requirements.txt                # Pinned dependencies with PyTorch index configuration
├── run.py                          # Development Flask server launcher
├── process_dataset.py              # CLI entry point for data ingestion, validation & splitting
├── README.md                       # Comprehensive platform documentation
├── docs/
│   └── api_guide.md                # Comprehensive REST API & SQLite Persistence Guide
├── config/
│   └── market_data_nse.json        # Central market data configuration for NSE equities
├── backend/
│   └── app/
│       ├── __init__.py             # Flask application factory (create_app with db, CORS & API blueprints)
│       ├── config.py               # Config classes (Development, Testing, Production)
│       ├── db.py                   # SQLAlchemy instance, SQLite pragmas, and serialization helpers
│       ├── data/                   # Phase 2 Market Data Pipeline Package
│       ├── features/               # Phase 3 Feature Engineering Package
│       ├── simulation/             # Phase 4 Portfolio Backtesting & Execution Engine
│       ├── rl/                     # Phase 5 & 6 Gymnasium Environment & PPO RL Engine
│       ├── risk/                   # Phase 7 Risk Management Gateway
│       ├── regime/                 # Phase 8 Market-Regime Clustering Engine
│       ├── realtime/               # Phase 11A Real-Time Market Data Integration
│       │   ├── __init__.py         # Package exports
│       │   ├── models.py           # Quote, connection states & config models
│       │   ├── validator.py        # Real-time tick validation & staleness detection
│       │   ├── base_provider.py    # Abstract base provider interface
│       │   ├── kite_provider.py    # Zerodha Kite Connect WebSocket provider
│       │   ├── mock_provider.py    # Mock/simulation provider for offline testing
│       │   └── manager.py          # Portfolio-aware subscription coordinator
│       ├── models/                 # SQLAlchemy Persistence Models
│       │   ├── __init__.py         # Model package exports
│       │   ├── asset.py            # Asset model
│       │   ├── backtest.py         # BacktestRun, TradeRecord, PortfolioSnapshotRecord
│       │   ├── agent.py            # AgentRun model
│       │   └── paper_trading.py    # PaperPortfolio, PortfolioBucketItem, PaperPosition, PaperTrade, PaperSnapshotRecord, RiskEventRecord
│       ├── services/               # Phase 9 Business Logic & Integration Services
│       │   ├── __init__.py         # Service package exports
│       │   ├── market_data_service.py # Validated OHLCV data query service
│       │   ├── backtest_service.py # Backtest runner & bounded ThreadPool background worker
│       │   ├── risk_service.py     # Pre-trade risk constraint evaluation service
│       │   ├── regime_service.py   # Regime detector artifact inspection & inference
│       │   ├── agent_service.py    # PPO agent artifact inspector & inference decision service
│       │   └── trading_pipeline_service.py # Multi-stock portfolios, stock bucket, rebalancing & simulation pipeline
│       └── routes/                 # Blueprint Route Handlers
│           ├── __init__.py         # Blueprint package exports
│           ├── main.py             # Landing page route (/)
│           ├── health.py           # Legacy health-check routes (/health, /api/health)
│           └── api/                # Phase 9 Institutional REST API (/api/v1/)
│               ├── __init__.py     # API v1 blueprint, response envelopes, global error handlers
│               ├── health.py       # GET /health, GET /system/status
│               ├── assets.py       # GET /assets, GET /assets/<sym>, GET /assets/<sym>/history
│               ├── dashboard.py    # GET /dashboard/summary
│               ├── backtests.py    # POST /backtests, GET /backtests, trades, equity-curve, metrics
│               ├── agent.py        # GET /agent/status, GET /agent/metrics, POST /agent/decisions
│               ├── risk.py         # GET /risk/config, GET /risk/status, POST /risk/evaluate
│               └── regime.py       # GET /regimes/<sym>, summary, latest
├── data/
│   ├── raw/                        # Untouched raw downloaded & uploaded CSVs (.gitkeep)
│   ├── processed/                  # Normalized, validated datasets (.gitkeep)
│   └── metadata/                   # Provenance JSON audit trails (.gitkeep)
├── instance/                       # Local SQLite storage (trademind.db)
├── models/                         # ML weights, PyTorch checkpoints, RL policies (.gitkeep)
├── notebooks/                      # Jupyter research & exploratory analysis notebooks (.gitkeep)
├── static/                         # Static CSS/JS assets for landing page
├── templates/                      # Jinja2 HTML5 templates
└── tests/
    ├── __init__.py                 # Test package initialization
    ├── conftest.py                 # Pytest fixtures (client, app)
    ├── test_api_v1.py              # Phase 9 REST API & SQLite integration test suite (26 tests)
    ├── test_data_pipeline.py       # Ingestion and validation unit tests
    ├── test_feature_engineering.py # Technical indicators and feature engineering tests
    ├── test_health.py              # Flask route and asset tests
    ├── test_nse_pipeline.py        # Central config, NSE download, refresh, and split tests
    ├── test_regime_detection.py    # Market-regime clustering and detector tests
    ├── test_risk_management.py     # Risk management and constraint enforcement tests
    ├── test_rl_env.py              # Gymnasium trading environment tests
    ├── test_rl_training.py         # PPO policy training pipeline tests
    └── test_simulation.py          # Portfolio simulator and baseline strategy tests
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

## 🚀 Phase 9: Flask REST API & SQLite Persistence

TradeMind exposes an institutional-grade REST API prefixed with `/api/v1/`, backed by SQLite persistence (`instance/trademind.db`) via Flask-SQLAlchemy, and equipped with a bounded in-process background worker for quantitative backtests.

### Summary of API Endpoints

- **System & Health**:
  - `GET /api/v1/health`: API status and database connectivity.
  - `GET /api/v1/system/status`: Platform modular capabilities (market data, strategies, PPO, risk, regime).
- **Asset Management**:
  - `GET /api/v1/assets`: List seeded canonical assets (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `ICICIBANK.NS`, `SBIN.NS`).
  - `GET /api/v1/assets/<symbol>`: Single asset details and data summary.
  - `GET /api/v1/assets/<symbol>/history`: Validated historical OHLCV bars (`start_date`, `end_date`, `limit`).
- **Dashboard Overview**:
  - `GET /api/v1/dashboard/summary`: Aggregated dashboard summary assembled from actual persisted records (zero fake metrics).
- **Quantitative Backtesting**:
  - `POST /api/v1/backtests`: Create and queue a backtest (`buy_and_hold`, `ma_crossover`). Returns HTTP `202 Accepted` (or `201 Created` with `?sync=true`).
  - `GET /api/v1/backtests`: Paginated listing of backtests with filtering.
  - `GET /api/v1/backtests/<id>`: Backtest metadata, status, and metrics.
  - `GET /api/v1/backtests/<id>/trades`: Paginated executed trades.
  - `GET /api/v1/backtests/<id>/equity-curve`: Daily mark-to-market valuation series.
  - `GET /api/v1/backtests/<id>/metrics`: Institutional performance metrics (Sharpe, CAGR, Drawdown, Win Rate, Turnover).
- **PPO Reinforcement Learning**:
  - `GET /api/v1/agent/status`: Model installation, artifact availability, and inference readiness.
  - `GET /api/v1/agent/metrics`: Offline evaluation benchmark metrics.
  - `POST /api/v1/agent/decisions`: Deterministic PPO policy action inference without automatic training (503 if model missing).
- **Risk Management Gateway**:
  - `GET /api/v1/risk/config`: Active exposure, drawdown, and cooldown thresholds.
  - `GET /api/v1/risk/status`: Risk gateway status (research mode note).
  - `POST /api/v1/risk/evaluate`: Pre-trade order inspection against constraints.
- **Market-Regime Detection**:
  - `GET /api/v1/regimes/<symbol>`: Fitted regime detector metadata.
  - `GET /api/v1/regimes/<symbol>/summary`: Empirical cluster summaries and descriptive labels.
  - `GET /api/v1/regimes/<symbol>/latest`: Retrospective regime classification for latest bar.

For detailed request/response schemas, error handling codes, and parameter bounds, see [`docs/api_guide.md`](file:///c:/TradeMind/docs/api_guide.md).

---

## 🏛️ Phase 9.5B: Persistent Storage, Database Integration & End-to-End System Integration

Phase 9.5B completes the persistent, audited, recoverable, and end-to-end integration across all TradeMind components.

### 1. Unified End-to-End Execution Flow

$$\text{Historical Data} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Market Regime Detection} \longrightarrow \text{PPO Policy Inference} \longrightarrow \text{Risk Management Gate} \longrightarrow \text{Portfolio Simulator Execution} \longrightarrow \text{SQLite Persistence} \longrightarrow \text{REST API} \longrightarrow \text{Frontend Dashboard}$$

- **Zero Future Leakage**: Observations are constructed using only data available up to decision timestamp $t$.
- **Protective Risk Gateway**: Every trade is inspected by `RiskManager` before ledger execution. Interventions are recorded in `RiskEventRecord`.
- **Atomic Database Transactions**: Paper trades, updated position ledgers, and audit events are committed together; failures trigger complete transaction rollback.

### 2. Multi-Tier Persistence Architecture

| Data Domain | Storage Mechanism | Schema / Structure | Recovery & Integrity Protection |
| :--- | :--- | :--- | :--- |
| **Relational Metadata & Simulation State** | SQLite (`instance/trademind.db`) | `Asset`, `BacktestRun`, `TradeRecord`, `RegisteredModel`, `PaperPortfolio`, `PaperPosition`, `PaperTrade`, `RiskEventRecord` | Foreign keys, cascading constraints, WAL mode, online atomic backup API (`sqlite3.backup`) |
| **Market Data (Raw & Processed)** | Local filesystem (`data/raw/`, `data/processed/`, `data/metadata/`) | Normalized OHLCV CSVs (`{symbol}_{freq}_processed.csv`) + JSON provenance reports | SHA-256 checksums, atomic temporary file replacement, failure rollback on error |
| **Model Artifacts & Checkpoints** | Configurable directory (`models/checkpoints/`) | SB3 PPO `.zip` policies + `training_metadata.json` | Model Registry table tracking candidate/approved/active states, SHA-256 validation, safe rollback |

### 3. Phase 9.5B REST API Additions

#### A. Model Registry Management (`/api/v1/models`)
- `GET /api/v1/models`: List registered models with optional `?symbol=` and `?status=` filters.
- `GET /api/v1/models/<id>`: Retrieve model metadata, observation space dimension, and validation metrics.
- `POST /api/v1/models/sync`: Auto-discover unindexed checkpoints in `models/` and register them.
- `POST /api/v1/models/<id>/approve`: Promote candidate model to approved status.
- `POST /api/v1/models/<id>/activate`: Set model as active inference policy for its symbol (deactivating prior active policy).
- `POST /api/v1/models/<symbol>/rollback`: Roll back active model to the previously approved checkpoint.

#### B. Paper Trading Simulation, Stock Bucket & Multi-Stock Portfolios (`/api/v1/trading`)
- `GET /api/v1/trading/portfolios`: List all configured portfolios with performance metrics and target allocations.
- `POST /api/v1/trading/portfolios`: Create an isolated paper portfolio configured from the Stock Bucket/Cart.
- `GET /api/v1/trading/portfolios/<name_or_id>`: Retrieve specific portfolio balances, bucket items, and active positions.
- `PUT /api/v1/trading/portfolios/<name_or_id>/bucket`: Update stock bucket selection and target allocation weights.
- `POST /api/v1/trading/portfolios/<name_or_id>/rebalance`: Execute simulated multi-asset rebalancing across bucket stocks.
- `GET /api/v1/trading/portfolios/<name_or_id>/snapshots`: Valuation snapshots for historical equity curve rendering.
- `DELETE /api/v1/trading/portfolios/<name_or_id>`: Delete a custom portfolio (default portfolio protected).
- `POST /api/v1/trading/step`: Execute a deterministic simulation step (Data → Features → Regime → PPO → Risk Gate → Persistence).
- `GET /api/v1/trading/portfolio`: Get current paper portfolio state (cash, positions, total value, peak value, drawdown).
- `GET /api/v1/trading/trades`: Paginated list of simulated executed trades with fee and slippage audit.
- `GET /api/v1/trading/risk-events`: Paginated list of institutional risk interventions (blocked or constrained orders).

#### C. Incremental Market Data Updates (`/api/v1/assets`)
- `GET /api/v1/assets/<symbol>/metadata`: Return on-disk row count, date ranges, file path, and SHA-256 checksum.
- `POST /api/v1/assets/refresh`: Trigger atomic incremental update for a single symbol or all configured symbols.

### 4. Online SQLite Backup & Disaster Recovery Utility

A dedicated online backup and restore utility is provided in `scripts/backup_restore.py`:

```powershell
# 1. Create a point-in-time consistent backup (DB + Market Data + Models)
.\.venv\Scripts\python.exe scripts/backup_restore.py backup --output-dir backups

# 2. Verify archive integrity and view contents
.\.venv\Scripts\python.exe scripts/backup_restore.py list

# 3. Restore from a backup archive with automatic pre-restore safety snapshot
.\.venv\Scripts\python.exe scripts/backup_restore.py restore --archive backups/trademind_backup_YYYYMMDD_HHMMSS.tar.gz
```

### 5. Deployment Persistence Configuration

For production deployment (e.g. Docker, Kubernetes, Cloud Run), mount persistent volumes (PV/PVC) to preserve the following stateful directories:

| Mount Path | Volume Type | Minimum Recommended Size | Required Survival Across Restarts |
| :--- | :--- | :--- | :--- |
| `/app/instance` | Persistent Volume | 5 GB | **Critical** (Application database, portfolio states, model registry) |
| `/app/data/processed` | Persistent Volume | 10 GB | **Critical** (Validated historical OHLCV datasets) |
| `/app/data/metadata` | Persistent Volume | 1 GB | **Critical** (Data provenance and validation audit trails) |
| `/app/models` | Persistent Volume | 10 GB | **Critical** (Trained PyTorch weights and SB3 PPO policy artifacts) |
| `/app/data/raw` | Object Storage / PV | 20 GB | **Recommended** (Immutable original provider downloads) |

**Environment Variables**:
- `DATABASE_URL`: Defaults to `sqlite:///instance/trademind.db`. Supports PostgreSQL (`postgresql://user:pass@host:5432/trademind`) without breaking local workflows.
- `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `MODELS_DIR`: Override filesystem storage paths.
- `MAX_BACKTEST_WORKERS`: Bounded thread concurrency limit (default: 2).
- `CORS_ORIGINS`: Allowed client origins for REST API access.

---

## ⚡ Phase 11: Real-Time Market Data & WebSocket Subscription Management

TradeMind AI incorporates a modular real-time streaming market data subsystem (Phases 11A & 11B) for Indian NSE equities using the Zerodha Kite Connect WebSocket protocol (`KiteTicker`) with fallback to an offline deterministic mock provider:

### Key Capabilities:
- **Connection Lifecycle**: Full state tracking (`DISCONNECTED`, `CONNECTING`, `CONNECTED`, `RECONNECTING`, `AUTH_FAILED`, `ERROR`, `STALE`) with protection against concurrent duplicate connections.
- **Protocol Decoding**: Official Kite Connect binary packet decoder (`KiteProtocolParser`) supporting 8-byte LTP, 44-byte Quote, and 184-byte Full modes with 5-depth bid/ask books and exchange timestamps.
- **Dynamic Subscription Management**: Instrument mapping, incremental subscribe/unsubscribe, capacity enforcement, broker acknowledgement tracking, and portfolio-aware syncing (`to_add`, `to_remove` deltas without reconnecting).
- **Automatic Reconnection & Reconciliation**: Bounded exponential backoff retry loop with automatic resubscription of all desired instruments upon reconnect.
- **Granular Activity Diagnostics**: Distinguishes between inactive instruments (liquid market absence during open connection) and broken socket connections (`is_stale`).
- **Zero Credential Exposure**: In-memory credential isolation; credentials and tokens are strictly filtered out of API responses, frontend payloads, and log entries.

---

## 🧠 Phase 12: Real-Time RL Policy Trading Signals

TradeMind AI evaluates live completed market candles through the trained PPO policy model (`RLSignalService`) to generate deterministic trading signals (`BUY`, `SELL`, `HOLD`) with rigorous timeframe protection and pre-trade risk gating:
- **Model & Timeframe Safety**: Validates input observations against the canonical 12-dimensional schema. Since the PPO model was trained on daily (`1d`) observations, intraday requests (`1m`, `5m`) are strictly classified as `TIMEFRAME_MISMATCH` with zero trades generated.
- **Pre-Trade Risk Engine Gating**: Orders are inspected against institutional risk constraints (drawdown, max exposure, allocation limits). High-risk signals are classified as `RISK_BLOCKED`.
- **Comprehensive Audit Trail**: Persists the full 12-dimensional observation vector, normalized features, and risk intervention details in `rl_trading_signals`.

---

## 📈 Phase 13: Automated Paper Trading Engine

Phase 13 implements a reliable automated paper-trading execution engine (`PaperTradingService`) that consumes validated PPO signals and executes simulated trades against isolated virtual portfolios:
- **Strict Simulation Boundary**: Strictly NO real-money order routing or broker API connectivity.
- **Idempotency & Duplicate Prevention**: Composite tracking `(portfolio_id, signal_id)` ensures the exact same signal can never execute multiple times.
- **Multi-Portfolio Isolation**: Virtual portfolios maintain independent initial capital, available cash, positions, trade history, peak equity, and risk limits.
- **Institutional Transaction Cost Modeling**: Configurable slippage (default 5 bps, 0.05%) and commission fees (default 3 bps, 0.03%) with accurate weighted average cost basis accounting.
- **Pre-Trade Risk Gateway**: Real-time integration with `RiskManager` blocking orders when portfolio drawdown exceeds configured limits.
- **Per-Portfolio Automation Toggle**: Ability to start and pause automated paper trading per portfolio (`/start`, `/stop`, `is_active`).
- **Signal-to-Trade Audit Lineage**: Complete bidirectional traceability between generated `RLTradingSignal` and resulting `PaperTrade` records.

---

## ⚠️ Data Limitations & Licensing Considerations

1. **Development Source**: Yahoo Finance (`yfinance`) is used as an initial development and research data provider. It is **not** an official NSE data feed.
2. **Official NSE Data Integration Point**: For production backtesting or institutional trade simulation, users can drop official Bhavcopy CSV files or licensed vendor feeds into `data/raw/` and process them through the same validation pipeline via `process_dataset.py --file <path>`.
3. **Trading Holidays & Gaps**: The pipeline detects gaps between consecutive trading sessions (e.g. weekends, Diwali, Republic Day) and logs them in [`ValidationReport.detected_gaps`](file:///c:/TradeMind/backend/app/data/models.py#L10-L24). It does **not** invent synthetic bars for non-trading sessions.
4. **No Live Execution**: TradeMind does not store brokerage API keys or route orders to exchanges.
5. **Background Worker Limitations**: Backtests use a local bounded ThreadPoolExecutor (default 2 workers). Jobs are tracked in SQLite. It is designed for single-node development and research rather than distributed durable job queues (e.g. Celery/Redis).
