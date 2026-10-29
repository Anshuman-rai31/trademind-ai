"""Unit and Integration Tests for TradeMind Phase 10A: Interactive Market Visualization.

Validates:
1. Historical OHLCV data query API for all 5 canonical NSE symbols.
2. Chronological sorting and time-series consistency (positive prices, high >= low, etc.).
3. Date range filtering and boundary conditions.
4. Technical indicator computations (SMA 5, SMA 20, RSI 14, MACD) without future-data leakage.
5. Error handling for invalid symbols, date ranges, and parameter limits.
6. Empty result handling when date range contains no trading sessions.
7. Frontend HTML templates, script loading, and UI element integration.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


# ==============================================================================
# 1. Historical Data Retrieval & Field Consistency Tests
# ==============================================================================

def test_phase_10a_historical_ohlcv_response(client):
    """Test that GET /api/v1/assets/<symbol>/history returns properly formatted OHLCV bars."""
    res = client.get("/api/v1/assets/RELIANCE.NS/history?limit=20")
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert "data" in body
    assert "meta" in body

    bars = body["data"]
    assert len(bars) == 20

    for bar in bars:
        for field in ("timestamp", "open", "high", "low", "close", "volume", "symbol"):
            assert field in bar
            assert bar[field] is not None

        # Positivity & relational price bounds
        assert bar["open"] > 0
        assert bar["high"] > 0
        assert bar["low"] > 0
        assert bar["close"] > 0
        assert bar["volume"] >= 0
        assert bar["high"] >= bar["low"]
        assert bar["high"] >= min(bar["open"], bar["close"])
        assert bar["low"] <= max(bar["open"], bar["close"])
        assert bar["symbol"] == "RELIANCE.NS"


def test_phase_10a_chronological_sorting(client):
    """Test that historical bars are returned in strict chronological ascending order."""
    res = client.get("/api/v1/assets/TCS.NS/history?limit=50")
    assert res.status_code == 200
    bars = res.get_json()["data"]
    assert len(bars) > 1

    timestamps = [b["timestamp"] for b in bars]
    sorted_timestamps = sorted(timestamps)
    assert timestamps == sorted_timestamps


def test_phase_10a_all_five_canonical_symbols(client):
    """Test that all 5 canonical NSE stocks have valid historical data available."""
    canonical_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]
    for sym in canonical_symbols:
        res = client.get(f"/api/v1/assets/{sym}/history?limit=10")
        assert res.status_code == 200, f"Failed for symbol {sym}"
        body = res.get_json()
        assert body["success"] is True
        assert len(body["data"]) == 10
        assert body["data"][0]["symbol"] == sym


# ==============================================================================
# 2. Date Range Filtering Tests
# ==============================================================================

def test_phase_10a_date_range_filtering(client):
    """Test that start_date and end_date filter bars correctly within inclusive bounds."""
    start_date = "2024-03-01"
    end_date = "2024-03-28"
    res = client.get(f"/api/v1/assets/INFY.NS/history?start_date={start_date}&end_date={end_date}&limit=100")
    assert res.status_code == 200
    bars = res.get_json()["data"]
    assert len(bars) > 0

    for b in bars:
        d = b["timestamp"].split("T")[0]
        assert start_date <= d <= end_date


def test_phase_10a_empty_date_range_returns_empty_list(client):
    """Test that a date range containing no trading days returns empty data with 200 OK."""
    # Weekend range or historic pre-start range
    res = client.get("/api/v1/assets/RELIANCE.NS/history?start_date=2010-01-01&end_date=2010-01-05")
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert body["data"] == []
    assert body["meta"]["returned_bars"] == 0


# ==============================================================================
# 3. Technical Indicator Calculation & Anti-Leakage Tests
# ==============================================================================

def test_phase_10a_technical_indicators_overlay(client):
    """Test that requesting indicators=true appends SMA(5,20), RSI(14), and MACD."""
    res = client.get("/api/v1/assets/ICICIBANK.NS/history?limit=60&indicators=true")
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    bars = body["data"]
    assert len(bars) == 60

    # Inspect the 50th bar (well past warm-up for all indicators)
    mature_bar = bars[45]
    for ind in ("sma_5", "sma_20", "rsi_14", "macd_line", "macd_signal", "macd_histogram"):
        assert ind in mature_bar
        assert mature_bar[ind] is not None
        assert isinstance(mature_bar[ind], (int, float))

    # RSI must be strictly in [0, 100]
    rsi_val = mature_bar["rsi_14"]
    assert 0.0 <= rsi_val <= 100.0

    # Moving averages must be within realistic price boundaries
    sma5_val = mature_bar["sma_5"]
    sma20_val = mature_bar["sma_20"]
    assert sma5_val > 0
    assert sma20_val > 0


def test_phase_10a_zero_future_leakage_in_indicators(client):
    """Verify that indicator value at bar t does not change when future bars are truncated."""
    # Query 80 bars
    res_full = client.get("/api/v1/assets/SBIN.NS/history?limit=80&indicators=true")
    assert res_full.status_code == 200
    bars_full = res_full.get_json()["data"]

    # Target bar index 50
    target_bar_full = bars_full[50]

    # Query only 51 bars (0 to 50, truncating all future bars from 51 onwards)
    res_trunc = client.get("/api/v1/assets/SBIN.NS/history?limit=51&indicators=true")
    assert res_trunc.status_code == 200
    bars_trunc = res_trunc.get_json()["data"]
    target_bar_trunc = bars_trunc[50]

    # Assert indicators at bar 50 are strictly identical
    assert target_bar_full["timestamp"] == target_bar_trunc["timestamp"]
    assert pytest.approx(target_bar_full["close"], abs=1e-5) == target_bar_trunc["close"]
    assert pytest.approx(target_bar_full["sma_5"], abs=1e-5) == target_bar_trunc["sma_5"]
    assert pytest.approx(target_bar_full["sma_20"], abs=1e-5) == target_bar_trunc["sma_20"]
    assert pytest.approx(target_bar_full["rsi_14"], abs=1e-5) == target_bar_trunc["rsi_14"]
    assert pytest.approx(target_bar_full["macd_line"], abs=1e-5) == target_bar_trunc["macd_line"]


# ==============================================================================
# 4. Error Handling & Validation Tests
# ==============================================================================

def test_phase_10a_invalid_symbol_handling(client):
    """Test that invalid symbols return 400 Bad Request."""
    res = client.get("/api/v1/assets/INVALID_TICKER!/history")
    assert res.status_code == 400
    body = res.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_SYMBOL"


def test_phase_10a_invalid_date_range_handling(client):
    """Test that start_date > end_date returns 400 with INVALID_DATE_RANGE."""
    res = client.get("/api/v1/assets/RELIANCE.NS/history?start_date=2024-05-10&end_date=2024-05-01")
    assert res.status_code == 400
    body = res.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_DATE_RANGE"


def test_phase_10a_invalid_limit_handling(client):
    """Test that limit <= 0 or limit > 5000 returns 400 with INVALID_PARAMETER."""
    res_zero = client.get("/api/v1/assets/RELIANCE.NS/history?limit=0")
    assert res_zero.status_code == 400
    assert res_zero.get_json()["error"]["code"] == "INVALID_PARAMETER"

    res_large = client.get("/api/v1/assets/RELIANCE.NS/history?limit=6000")
    assert res_large.status_code == 400
    assert res_large.get_json()["error"]["code"] == "INVALID_PARAMETER"


# ==============================================================================
# 5. Frontend Integration & UI Components Tests
# ==============================================================================

def test_phase_10a_frontend_elements_present(client):
    """Test that landing page template renders all Phase 10A chart controls and elements."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Core chart card and canvas
    assert 'id="market-chart-card"' in html
    assert 'id="market-candlestick-chart"' in html
    assert 'id="chart-symbol-select"' in html

    # Period preset buttons
    assert 'id="period-presets"' in html
    assert 'data-period="1M"' in html
    assert 'data-period="1Y"' in html
    assert 'data-period="ALL"' in html

    # Date range inputs
    assert 'id="chart-start-date"' in html
    assert 'id="chart-end-date"' in html
    assert 'id="btn-apply-dates"' in html

    # Technical indicator toggles
    assert 'id="toggle-sma5"' in html
    assert 'id="toggle-sma20"' in html
    assert 'id="toggle-rsi"' in html
    assert 'id="toggle-macd"' in html

    # Reset zoom & loading overlay
    assert 'id="btn-chart-reset-zoom"' in html
    assert 'id="chart-loading-overlay"' in html
    assert 'id="chart-empty-state"' in html
    assert 'id="chart-error-banner"' in html

    # Plotly script inclusion
    assert 'plotly.min.js' in html or 'plotly' in html
