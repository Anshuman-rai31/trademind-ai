# TradeMind REST API & SQLite Persistence Guide (Phase 9)

This guide documents the TradeMind Flask REST API, SQLite persistence layer, endpoint specifications, request/response formats, error envelopes, and integration interfaces.

---

## 1. Quickstart & Server Launch

### A. Environment & Dependencies

Install all core and Phase 9 dependencies inside your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### B. Database Initialization

Database schemas and initial canonical asset records (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `ICICIBANK.NS`, `SBIN.NS`) are automatically initialized upon startup. You can also explicitly verify initialization using Python:

```powershell
.\.venv\Scripts\python.exe -c "from backend.app import create_app; from backend.app.db import init_db; app = create_app('development'); init_db(app); print('Database initialized successfully.')"
```

The SQLite database file is located at `instance/trademind.db`.

### C. Launch Development Server

```powershell
.\.venv\Scripts\python.exe run.py
```

- Server URL: `http://127.0.0.1:5000/`
- API Base URL: `http://127.0.0.1:5000/api/v1/`
- Health Check: `http://127.0.0.1:5000/api/v1/health`

### D. Run Automated Tests

The complete offline test suite (all 206 unit and integration tests) runs in isolation without external network calls:

```powershell
.\.venv\Scripts\pytest.exe -v
```

To run Phase 9 API tests specifically:

```powershell
.\.venv\Scripts\pytest.exe tests/test_api_v1.py -v
```

---

## 2. API Architecture & Standard Response Envelope

All API endpoints are prefixed with `/api/v1/`.

### Successful Response Envelope (`HTTP 200 / 201 / 202`)

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "timestamp": "2026-10-28T16:00:00.000000Z"
  }
}
```

### Standardized Error Envelope (`HTTP 400 / 404 / 405 / 500 / 503`)

```json
{
  "success": false,
  "error": {
    "code": "DATA_UNAVAILABLE",
    "message": "Historical data is not available for symbol 'SBIN.NS'."
  },
  "meta": {
    "timestamp": "2026-10-28T16:00:00.000000Z"
  }
}
```

---

## 3. Endpoints Reference Table

| HTTP Method | Endpoint Path | Description | Key Parameters |
|---|---|---|---|
| `GET` | `/api/v1/health` | Health check & database connection | None |
| `GET` | `/api/v1/system/status` | Modular platform capability breakdown | None |
| `GET` | `/api/v1/assets` | List supported trading assets | None |
| `GET` | `/api/v1/assets/<symbol>` | Get details & data summary for single asset | `symbol` (in path) |
| `GET` | `/api/v1/assets/<symbol>/history` | Query validated historical OHLCV bars | `start_date`, `end_date`, `limit` (max 5000) |
| `GET` | `/api/v1/dashboard/summary` | Aggregated dashboard overview metrics | None |
| `POST` | `/api/v1/backtests` | Queue backtest execution (Buy&Hold, MACrossover) | `name`, `symbol`, `strategy_type`, `initial_capital`, `parameters`, `sync` |
| `GET` | `/api/v1/backtests` | List backtests with pagination and filtering | `page`, `limit`, `symbol`, `status` |
| `GET` | `/api/v1/backtests/<id>` | Retrieve metadata and computed metrics for backtest | `id` (in path) |
| `GET` | `/api/v1/backtests/<id>/trades` | Paginated executed order fills for backtest | `id` (in path), `page`, `limit` |
| `GET` | `/api/v1/backtests/<id>/equity-curve` | Mark-to-market daily portfolio valuation series | `id` (in path) |
| `GET` | `/api/v1/backtests/<id>/metrics` | Detailed computed institutional risk metrics | `id` (in path) |
| `GET` | `/api/v1/agent/status` | PPO agent installation & model readiness | `symbol` (query, optional) |
| `GET` | `/api/v1/agent/metrics` | Latest persisted PPO policy evaluation metrics | `symbol` (query, optional) |
| `POST` | `/api/v1/agent/decisions` | Deterministic PPO policy action inference | `symbol`, `state` (features vector) |
| `GET` | `/api/v1/risk/config` | Active quantitative risk limits & parameters | None |
| `GET` | `/api/v1/risk/status` | Operational risk gateway status | None |
| `POST` | `/api/v1/risk/evaluate` | Pre-trade order inspection against constraints | `symbol`, `side`, `quantity`, `price`, `portfolio` |
| `GET` | `/api/v1/regimes/<symbol>` | Fitted regime detector configuration & metadata | `symbol` (in path) |
| `GET` | `/api/v1/regimes/<symbol>/summary` | Empirical cluster summaries & labels | `symbol` (in path) |
| `GET` | `/api/v1/regimes/<symbol>/latest` | Retrospective classification of latest bar | `symbol` (in path) |

---

## 4. Detailed Endpoint Usage & Examples

### A. System & Health

#### `GET /api/v1/health`
Returns system status and SQLite database connection test.

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "app_name": "TradeMind",
    "version": "0.1.0",
    "environment": "development",
    "database": {
      "connected": true,
      "engine": "sqlite"
    }
  },
  "meta": { "timestamp": "2026-10-28T16:00:00Z" }
}
```

#### `GET /api/v1/system/status`
Returns status of market data, baseline strategies, PPO agent, risk management, and market regime detection.

---

### B. Asset Management

#### `GET /api/v1/assets`
Returns all canonical supported equities.

#### `GET /api/v1/assets/RELIANCE.NS/history?limit=3`
Fetches validated OHLCV records from `data/processed/`.

```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2023-12-31T18:30:00Z",
      "open": 1274.9,
      "high": 1287.89,
      "low": 1271.24,
      "close": 1279.69,
      "volume": 4030540,
      "symbol": "RELIANCE.NS"
    }
  ],
  "meta": {
    "returned_bars": 1,
    "total_matched_bars": 10,
    "limit": 3
  }
}
```

---

### C. Backtesting

#### `POST /api/v1/backtests`
Creates and queues a backtest simulation.

**Request Body:**
```json
{
  "name": "Reliance Moving Average Test",
  "symbol": "RELIANCE.NS",
  "strategy_type": "ma_crossover",
  "start_date": "2023-12-31",
  "end_date": "2024-01-11",
  "initial_capital": 100000.0,
  "parameters": {
    "fast_window": 2,
    "slow_window": 5,
    "allocation_pct": 0.90
  }
}
```

**Response (`HTTP 202 Accepted`):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Reliance Moving Average Test",
    "symbol": "RELIANCE.NS",
    "strategy_type": "ma_crossover",
    "status": "queued",
    "initial_capital": 100000.0,
    "created_at": "2026-10-28T16:00:00Z"
  }
}
```

*Note:* Use query parameter `?sync=true` for synchronous execution during automated testing.

#### `GET /api/v1/backtests/1/equity-curve`
Returns chronological portfolio NAV mark-to-market daily snapshots:
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-01 18:30:00+00:00",
      "cash": 99961.43,
      "holdings_value": 0.0,
      "portfolio_value": 99961.43,
      "drawdown": 0.00038
    }
  ]
}
```

---

### D. PPO Agent

#### `POST /api/v1/agent/decisions`
Generates a deterministic research decision using a trained PPO policy without modifying weights or launching automatic training.

**Request Body:**
```json
{
  "symbol": "RELIANCE.NS",
  "state": [0.012, 0.024, -0.005, 54.2, 0.018]
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "success": true,
  "data": {
    "symbol": "RELIANCE.NS",
    "action": 1,
    "action_name": "BUY",
    "model_artifact": "best_model.zip",
    "disclaimer": "This decision is an experimental model output generated for quantitative research and simulation purposes only. It does not constitute financial advice or a guarantee of future price performance."
  }
}
```

*Error behavior:* If no trained model artifact exists on disk, returns `HTTP 503 Service Unavailable` with code `MODEL_UNAVAILABLE`.

---

### E. Risk Management

#### `POST /api/v1/risk/evaluate`
Inspects proposed trading orders before execution.

**Request Body:**
```json
{
  "symbol": "RELIANCE.NS",
  "side": "BUY",
  "quantity": 1000.0,
  "price": 1280.0,
  "portfolio": {
    "cash": 100000.0,
    "holdings": {}
  }
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "success": true,
  "data": {
    "requested_action": {
      "symbol": "RELIANCE.NS",
      "side": "BUY",
      "quantity": 1000.0,
      "price": 1280.0,
      "estimated_value": 1280000.0
    },
    "effective_action": null,
    "approved": false,
    "intervention_occurred": true,
    "intervention_reason": "INSUFFICIENT_CASH",
    "intervention_status": "REJECTED",
    "details": "Proposed order cost INR 1,280,000.00 exceeds available cash INR 100,000.00."
  }
}
```

---

### F. Market-Regime Detection

#### `GET /api/v1/regimes/RELIANCE.NS/latest`
Returns the historical regime classification for the latest observation bar. Does not retrain detector.

```json
{
  "success": true,
  "data": {
    "symbol": "RELIANCE.NS",
    "timestamp": "2024-01-11T18:30:00Z",
    "regime_cluster": 0,
    "regime_label": "Regime 0 (Low Volatility, Bullish/Positive)",
    "disclaimer": "Market regime classifications are retrospective historical patterns derived from unsupervised clustering. They do not predict future returns."
  }
}
```

---

## 5. Configuration & Security

- **Environment Setting**: Set `FLASK_ENV=development` or `FLASK_ENV=production`.
- **Secret Key**: Set `SECRET_KEY` in environment variables for production deployments.
- **CORS**: Configure allowed origins via `CORS_ORIGINS="http://localhost:3000,http://localhost:5173"`. In development mode, `*` is permitted for local frontends.
- **Worker Concurrency**: Bounded local threads for backtests are controlled via `MAX_BACKTEST_WORKERS=2` (default 2).
- **SQLite Database Path**: Located at `instance/trademind.db`. Override via `DATABASE_URL`.
- **Model Storage**: Model artifacts are stored in `models/`. Models are never loaded into SQLite directly.
