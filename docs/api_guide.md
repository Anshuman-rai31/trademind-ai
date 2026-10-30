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
| `GET` | `/api/v1/assets/<symbol>/history` | Query validated historical OHLCV bars & optional technical indicators | `start_date`, `end_date`, `limit` (max 5000), `indicators` (boolean) |
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

#### `GET /api/v1/assets/RELIANCE.NS/history?limit=3&indicators=true`
Fetches validated OHLCV records from `data/processed/` with technical indicator overlays (`sma_5`, `sma_20`, `rsi_14`, `macd_line`, `macd_signal`, `macd_histogram`).

```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-02T18:30:00Z",
      "open": 1274.9,
      "high": 1287.89,
      "low": 1271.24,
      "close": 1279.69,
      "volume": 4030540,
      "symbol": "RELIANCE.NS",
      "sma_5": 1270.42,
      "sma_20": 1255.81,
      "rsi_14": 62.45,
      "macd_line": 8.35,
      "macd_signal": 6.12,
      "macd_histogram": 2.23
    }
  ],
  "meta": {
    "returned_bars": 1,
    "total_matched_bars": 2469,
    "limit": 3,
    "include_indicators": true
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

### G. User-Configured Multi-Stock Portfolios & Stock Bucket (Phase 9.5B+)

#### Portfolio Architecture & Security Note
> **Single-Tenant / Local Research Environment:** TradeMind currently runs in a single-tenant local quantitative research environment without user authentication accounts. A default `user_id="default_user"` is stored in the database models for future schema compatibility. Strict data isolation across portfolios is enforced at the `portfolio_id` and unique `name` database foreign-key boundary.

#### `GET /api/v1/trading/portfolios`
Lists all user-configured portfolios with summary balances, allocations, and performance metrics.

**Response (`HTTP 200 OK`):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "default",
      "user_id": "default_user",
      "initial_cash": 100000.0,
      "cash": 100000.0,
      "portfolio_value": 100000.0,
      "peak_value": 100000.0,
      "total_realized_pnl": 0.0,
      "total_unrealized_pnl": 0.0,
      "total_pnl": 0.0,
      "return_pct": 0.0,
      "drawdown_pct": 0.0,
      "allocation_method": "EQUAL_WEIGHT",
      "strategy_type": "BUY_AND_HOLD",
      "bucket": [
        {"id": 1, "symbol": "RELIANCE.NS", "target_weight": 0.2, "target_pct": 20.0},
        {"id": 2, "symbol": "TCS.NS", "target_weight": 0.2, "target_pct": 20.0},
        {"id": 3, "symbol": "INFY.NS", "target_weight": 0.2, "target_pct": 20.0},
        {"id": 4, "symbol": "ICICIBANK.NS", "target_weight": 0.2, "target_pct": 20.0},
        {"id": 5, "symbol": "SBIN.NS", "target_weight": 0.2, "target_pct": 20.0}
      ],
      "positions": []
    }
  ]
}
```

#### `POST /api/v1/trading/portfolios`
Creates an isolated paper-trading portfolio configured with selected stock bucket assets and target weights.

**Request Body:**
```json
{
  "name": "tech_balanced",
  "initial_cash": 200000.0,
  "allocation_method": "CUSTOM",
  "strategy_type": "BUY_AND_HOLD",
  "risk_config": { "max_drawdown_limit": 0.15 },
  "bucket": [
    {"symbol": "RELIANCE.NS", "target_weight": 0.40},
    {"symbol": "TCS.NS", "target_weight": 0.35},
    {"symbol": "INFY.NS", "target_weight": 0.25}
  ]
}
```

**Response (`HTTP 201 Created`):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "name": "tech_balanced",
    "initial_cash": 200000.0,
    "cash": 200000.0,
    "portfolio_value": 200000.0,
    "allocation_method": "CUSTOM",
    "bucket": [
      {"symbol": "RELIANCE.NS", "target_weight": 0.4},
      {"symbol": "TCS.NS", "target_weight": 0.35},
      {"symbol": "INFY.NS", "target_weight": 0.25}
    ],
    "positions": []
  }
}
```

*Validation Rules:*
- Name must be unique and non-empty.
- Bucket must contain at least 1 supported NSE asset (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `ICICIBANK.NS`, `SBIN.NS`).
- No duplicate symbols in the bucket.
- Adding a stock to the bucket does *not* create position holdings until execution/rebalancing.
- In `CUSTOM` mode, target weights must be strictly positive and sum to 100% (within 1.5% tolerance).

#### `GET /api/v1/trading/portfolios/<name_or_id>`
Retrieves the complete state of a portfolio, including cash, holdings value, target bucket items, open positions, and return metrics.

#### `PUT /api/v1/trading/portfolios/<name_or_id>/bucket`
Updates the stock selection bucket and target allocation weights of an existing portfolio.

#### `POST /api/v1/trading/portfolios/<name_or_id>/rebalance`
Executes simulated rebalancing across configured bucket stocks:
1. Computes total mark-to-market portfolio value.
2. Calculates target shares per stock: `Target_Shares = floor(Portfolio_Value * target_weight / price)`.
3. Sells excess shares first to release cash, applying 0.05% slippage and 0.03% transaction fees.
4. Buys required shares up to available cash.
5. Updates positions, cash balance, and logs trades and equity snapshots.

#### `GET /api/v1/trading/portfolios/<name_or_id>/snapshots`
Returns the sequential valuation snapshots for rendering the portfolio equity curve.

#### `DELETE /api/v1/trading/portfolios/<name_or_id>`
Deletes a custom portfolio. Deletion of the `default` portfolio is prevented.

---

### H. Real-Time Market Data Provider APIs (Phase 11A)

Provides modular integration with streaming market-data providers (Zerodha Kite Connect WebSocket protocol / Mock provider for offline testing and verification).

#### Provider Configuration & Security
> **Security Guarantee:** Real-time provider credentials (`KITE_API_KEY`, `KITE_ACCESS_TOKEN`, `KITE_API_SECRET`) are read strictly from environment variables. They are never hardcoded, never stored in the database, and never exposed in API responses or frontend payloads.

#### `GET /api/v1/realtime/status`
Returns connection lifecycle state, telemetry, staleness indicator, and sanitized configuration.

**Response (`HTTP 200 OK`):**
```json
{
  "success": true,
  "data": {
    "status": "CONNECTED",
    "provider": "kite",
    "is_stale": false,
    "telemetry": {
      "status": "CONNECTED",
      "provider_name": "kite",
      "connected_at": "2026-10-29T13:40:00Z",
      "last_message_at": "2026-10-29T13:40:05Z",
      "last_heartbeat_at": "2026-10-29T13:40:05Z",
      "messages_received": 1420,
      "messages_rejected": 0,
      "subscribed_symbols_count": 5,
      "reconnect_attempts": 0,
      "last_error": null,
      "is_stale": false
    },
    "config": {
      "provider_name": "kite",
      "has_credentials": true,
      "has_api_key": true,
      "has_access_token": true,
      "max_subscriptions": 50,
      "stale_timeout_seconds": 15.0,
      "reconnect_max_retries": 5
    }
  }
}
```

#### `GET /api/v1/realtime/health`
Returns granular health diagnostics distinguishing socket connection health from individual instrument inactivity (e.g., illiquid stocks or closed market hours vs broken WebSocket):
```json
{
  "success": true,
  "data": {
    "status": "CONNECTED",
    "provider": "kite",
    "connected": true,
    "is_stale": false,
    "last_heartbeat_at": "2026-10-29T14:15:00.123456+00:00",
    "last_message_at": "2026-10-29T14:15:00.123456+00:00",
    "messages_received": 1420,
    "messages_rejected": 0,
    "reconnect_attempts": 0,
    "last_error": null,
    "subscriptions": {
      "total_subscribed": 5,
      "active_instruments": 4,
      "inactive_market_instruments": 1,
      "capacity": 50
    }
  }
}
```

#### `POST /api/v1/realtime/connect`
Initiates connection to the market data feed with exponential backoff and auth verification.

#### `POST /api/v1/realtime/disconnect`
Gracefully closes the streaming connection and releases background listeners.

#### `POST /api/v1/realtime/reconnect`
Triggers an immediate clean disconnect, bounded backoff, and automatic subscription reconciliation:
```json
{
  "success": true,
  "data": {
    "reconnected": true,
    "status": "CONNECTED",
    "subscriptions": ["ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"]
  }
}
```

#### `GET /api/v1/realtime/subscriptions`
Returns the active subscription list and configured subscription capacity. Pass `?details=true` for full registry telemetry:
```json
{
  "success": true,
  "data": {
    "total_registered": 5,
    "subscribed_count": 2,
    "active_instruments": 2,
    "inactive_market_instruments": 0,
    "max_subscriptions": 50,
    "instruments": {
      "RELIANCE.NS": {
        "symbol": "RELIANCE.NS",
        "instrument_token": 738561,
        "state": "SUBSCRIBED",
        "activity_status": "ACTIVE",
        "last_price": 2850.5,
        "last_seen_at": "2026-10-29T14:15:00.000000+00:00",
        "ticks_count": 45,
        "error_message": null,
        "subscribed_at": "2026-10-29T14:10:00.000000+00:00"
      }
    }
  }
}
```

#### `POST /api/v1/realtime/subscribe`
Subscribes to updates for specified instruments:
```json
{
  "symbols": ["RELIANCE.NS", "TCS.NS"]
}
```

#### `POST /api/v1/realtime/unsubscribe`
Unsubscribes from specified instruments:
```json
{
  "symbols": ["TCS.NS"]
}
```

#### `POST /api/v1/realtime/sync-portfolio`
Synchronizes active subscriptions with the user's active portfolio stock bucket:
```json
{
  "portfolio": "tech_balanced"
}
```
*Computes delta (`to_add` and `to_remove`) without disconnecting or duplicating connections.*

#### `GET /api/v1/realtime/quotes`
Retrieves latest validated quotes across all subscribed instruments.

#### `GET /api/v1/realtime/quotes/<symbol>`
Retrieves the latest validated quote for a specific instrument.

#### `GET /api/v1/realtime/freshness/<symbol>`
Returns instrument freshness status (`FRESH`, `AGING`, `STALE`, `OFFLINE`), event timestamp, local processing timestamp, and latency in milliseconds.

#### `GET /api/v1/realtime/candles/<symbol>`
Retrieves recent completed and in-progress OHLCV candles:
- Parameters: `interval` (`1m`, `5m`, `1d`, default: `5m`), `limit` (default: 50, max: 500), `include_in_progress` (`true`/`false`, default: `true`).

#### `GET /api/v1/realtime/candles/<symbol>/latest`
Retrieves the latest completed/closed OHLCV candle for the specified interval.

#### `GET /api/v1/realtime/candles/<symbol>/in-progress`
Retrieves the currently accumulating in-progress candle for the specified interval.

#### `GET /api/v1/realtime/features/<symbol>`
Evaluates the PPO feature engineering pipeline. Returns the 12-dimensional observation vector if compatible, or reports the timeframe mismatch if an intraday interval is requested against the daily PPO model checkpoint.

#### `GET /api/v1/realtime/dashboard`
Returns the aggregated real-time dashboard state (connection telemetry, active quotes, per-stock freshness, recent candles, and PPO feature readiness) for responsive frontend polling without page reloads.

---

### I. Real-Time RL Policy Signals API (Phase 12)

Provides endpoints for monitoring model-generated BUY, SELL, and HOLD trading signals evaluated on completed market candles, inspecting full observation and risk audit trails, checking policy readiness, and triggering on-demand candle evaluations.

#### `GET /api/v1/signals/latest`
Retrieves the latest signal generated per symbol, or for a specific symbol:
- Parameters: `symbol` (optional, e.g. `RELIANCE.NS`)
- Response:
```json
{
  "success": true,
  "data": {
    "action": "BUY",
    "proposed_action": "BUY",
    "effective_action": "BUY",
    "close_price": 2850.50,
    "confidence": null,
    "data_freshness": "FRESH",
    "id": 1,
    "interval": "1d",
    "model_id": "ppo_best_model",
    "model_version": "1.0.0",
    "model_checksum": "8f3a...",
    "signal_id": "sig_a1b2c3d4e5f67890",
    "source_candle_timestamp": "2026-10-30T00:00:00+00:00",
    "status": "VALID",
    "symbol": "RELIANCE.NS",
    "timestamp": "2026-10-30T10:15:30.123456+00:00"
  }
}
```

#### `GET /api/v1/signals/history`
Retrieves paginated historical trading signals with optional filters:
- Parameters: `symbol`, `action` (`BUY`/`SELL`/`HOLD`), `status` (`VALID`/`RISK_BLOCKED`/`TIMEFRAME_MISMATCH`/`REJECTED`), `interval` (`1d`/`5m`/`1m`), `page` (default: 1), `page_size` (default: 20, max: 100).
- Response includes pagination metadata: `total`, `page`, `page_size`, `total_pages`.

#### `GET /api/v1/signals/<id_or_signal_id>`
Retrieves full audit details of a specific signal:
- Includes the 12-dimensional observation vector `observation`
- Normalized feature breakdown dictionary `features`
- Pre-trade risk manager intervention evaluation `risk_evaluation`

#### `GET /api/v1/signals/health`
Returns signal generation engine readiness, active models loaded per symbol, supported symbols, and telemetry counters (`total_signals_generated`, `total_valid_signals`, `total_risk_blocked`, `total_timeframe_mismatches`, `total_duplicates_prevented`).

#### `POST /api/v1/signals/generate`
Triggers on-demand evaluation of a completed candle:
```json
{
  "symbol": "RELIANCE.NS",
  "interval": "1d"
}
```
*If `candle` object is omitted, fetches the latest completed bar from historical/persistent storage automatically.*

---

### J. Automated Paper Trading Engine API (Phase 13)

Provides endpoints to manage automated simulated trade execution against virtual portfolios, view current simulated positions and trades, inspect signal-to-trade lineage, start/pause paper trading per portfolio, and inspect execution engine health and transaction cost parameters.

#### `POST /api/v1/trading/portfolios/<name_or_id>/start`
Enables automated paper trading for the specified portfolio:
- Response includes updated portfolio configuration with `is_active: true`.

#### `POST /api/v1/trading/portfolios/<name_or_id>/stop`
Pauses automated paper trading for the specified portfolio:
- Prevents incoming PPO signals from generating trades against this portfolio (`PORTFOLIO_PAUSED`).

#### `GET /api/v1/trading/portfolios/<name_or_id>/status`
Returns paper trading status, available cash, portfolio NAV, active positions count, and total executed trades count.

#### `GET /api/v1/trading/portfolios/<name_or_id>/positions`
Returns current simulated positions, held quantities, average entry prices, current prices, market values, and unrealized P&L.

#### `GET /api/v1/trading/portfolios/<name_or_id>/trades`
Returns chronological simulated trade execution history, fill prices with slippage, institutional fees deducted, and source `signal_id` linkage.

#### `GET /api/v1/trading/signals/<signal_id>/trade`
Performs signal-to-trade audit lookup:
- Retrieves the specific `PaperTrade` record generated by a given `signal_id`.

#### `POST /api/v1/trading/execute-signal`
Executes an eligible PPO trading signal against a target paper portfolio:
```json
{
  "signal_id": "sig_a1b2c3d4e5f67890",
  "portfolio": "default"
}
```
*Applies pre-trade risk checks, slippage, institutional transaction fees (0.03%), and updates positions and cash atomically.*

#### `GET /api/v1/trading/engine/health`
Returns paper-trading execution engine health, transaction cost assumptions (fee rate: 0.03% / 3 bps, slippage rate: 0.05% / 5 bps), and operational telemetry counters (total signals, executed trades, duplicate attempts prevented, risk rejections, timeframe mismatches).

---

## 5. Configuration & Security

- **Environment Setting**: Set `FLASK_ENV=development` or `FLASK_ENV=production`.
- **Secret Key**: Set `SECRET_KEY` in environment variables for production deployments.
- **CORS**: Configure allowed origins via `CORS_ORIGINS="http://localhost:3000,http://localhost:5173"`. In development mode, `*` is permitted for local frontends.
- **Worker Concurrency**: Bounded local threads for backtests are controlled via `MAX_BACKTEST_WORKERS=2` (default 2).
- **SQLite Database Path**: Located at `instance/trademind.db`. Override via `DATABASE_URL`.
- **Model Storage**: Model artifacts are stored in `models/`. Models are never loaded into SQLite directly.


