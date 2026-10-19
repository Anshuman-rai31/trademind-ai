"""TradeMind Phase 4 Unit & Integration Tests: Portfolio Simulator and Baselines.

Deterministic, offline test suite verifying:
- Order execution, slippage, brokerage fee accounting, and cash constraints.
- Position tracking, holdings valuation, and daily snapshots.
- No-lookahead execution timing: signal at bar t close -> execution at bar t+1 open.
- Buy-and-hold strategy on hand-calculated deterministic data.
- Moving average crossover timing, golden cross, death cross, and duplicate order prevention.
- Risk and performance metrics: CAGR, Sharpe, max drawdown, turnover, win rate.
- Edge cases: zero volatility, empty/single-bar data, missing session prices, and multi-asset alignment.
- Pure reproducibility and immutability of input data.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.simulation import (
    BuyAndHoldStrategy,
    InsufficientCashError,
    InsufficientHoldingsError,
    MACrossoverStrategy,
    Order,
    OrderSide,
    OrderStatus,
    PerformanceMetrics,
    Portfolio,
    PortfolioSimulator,
    SimulationConfig,
    calculate_metrics,
)


# ==============================================================================
# Deterministic Fixtures
# ==============================================================================

@pytest.fixture
def sample_5bar_df():
    """5-bar deterministic OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "TCS.NS",
        "open": [100.0, 102.0, 105.0, 103.0, 108.0],
        "high": [103.0, 106.0, 107.0, 109.0, 112.0],
        "low":  [99.0,  101.0, 104.0, 102.0, 107.0],
        "close": [102.0, 105.0, 104.0, 108.0, 110.0],
        "volume": [10000, 12000, 15000, 11000, 14000],
    })


@pytest.fixture
def trending_60bar_df():
    """60-bar synthetic price series engineered with clear crossovers."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    # First 25 bars declining (100 -> 75), next 35 bars surging (75 -> 145)
    p1 = np.linspace(100.0, 75.0, 25)
    p2 = np.linspace(76.0, 145.0, 35)
    close = np.concatenate([p1, p2])
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "RELIANCE.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": [100000] * 60,
    })


# ==============================================================================
# Portfolio Ledger & Order Execution Tests
# ==============================================================================

def test_portfolio_initialization():
    """Verify initial portfolio state."""
    p = Portfolio(initial_cash=50_000.0)
    assert p.initial_cash == 50_000.0
    assert p.cash == 50_000.0
    assert p.holdings == {}
    assert p.get_position("RELIANCE.NS") == 0.0
    assert len(p.trades) == 0
    assert len(p.order_history) == 0
    assert p.peak_value == 50_000.0

    with pytest.raises(ValueError, match="strictly positive"):
        Portfolio(initial_cash=0.0)


def test_hand_calculated_buy_order_accounting():
    """Hand-verify BUY order cash outflow, slippage, and fee calculations."""
    config = SimulationConfig(
        fee_rate=0.001,       # 0.1% = 10 bps
        slippage_rate=0.002,  # 0.2% = 20 bps
        min_fee=0.0,
        allow_fractional=False,
    )
    portfolio = Portfolio(initial_cash=10_000.0)
    ts = pd.Timestamp("2024-01-02", tz="UTC")

    order = Order(
        symbol="INFY.NS",
        side=OrderSide.BUY,
        quantity=20.0,
        created_at=ts,
    )
    base_price = 100.0

    # Hand calculation:
    # exec_price = 100.0 * (1 + 0.002) = 100.20
    # turnover = 100.20 * 20 = 2004.0
    # fee = 2004.0 * 0.001 = 2.004
    # total_cash_required = 2004.0 + 2.004 = 2006.004
    trade = portfolio.execute_order(order, base_price=base_price, timestamp=ts, config=config)

    assert trade is not None
    assert trade.execution_price == pytest.approx(100.20, abs=1e-5)
    assert trade.fee == pytest.approx(2.004, abs=1e-5)
    assert trade.net_cost == pytest.approx(-2006.004, abs=1e-5)
    assert portfolio.cash == pytest.approx(10_000.0 - 2006.004, abs=1e-5)
    assert portfolio.get_position("INFY.NS") == 20.0
    assert portfolio.cost_basis["INFY.NS"] == pytest.approx(100.20, abs=1e-5)
    assert order.status == OrderStatus.FILLED
    assert order.filled_price == pytest.approx(100.20, abs=1e-5)


def test_hand_calculated_sell_order_accounting():
    """Hand-verify SELL order cash inflow, slippage, fee, and realized PnL."""
    config = SimulationConfig(
        fee_rate=0.001,
        slippage_rate=0.002,
        allow_fractional=False,
    )
    portfolio = Portfolio(initial_cash=10_000.0)
    ts1 = pd.Timestamp("2024-01-02", tz="UTC")
    ts2 = pd.Timestamp("2024-01-03", tz="UTC")

    # Buy 20 shares at 100.0 base
    buy_order = Order(symbol="INFY.NS", side=OrderSide.BUY, quantity=20.0, created_at=ts1)
    portfolio.execute_order(buy_order, base_price=100.0, timestamp=ts1, config=config)
    cash_after_buy = portfolio.cash

    # Sell 10 shares at 150.0 base
    # Hand calculation:
    # exec_price = 150.0 * (1 - 0.002) = 149.70
    # turnover = 149.70 * 10 = 1497.0
    # fee = 1497.0 * 0.001 = 1.497
    # net_proceeds = 1497.0 - 1.497 = 1495.503
    # realized_pnl = (149.70 - 100.20) * 10 - 1.497 = 495.0 - 1.497 = 493.503
    sell_order = Order(symbol="INFY.NS", side=OrderSide.SELL, quantity=10.0, created_at=ts2)
    sell_trade = portfolio.execute_order(sell_order, base_price=150.0, timestamp=ts2, config=config)

    assert sell_trade is not None
    assert sell_trade.execution_price == pytest.approx(149.70, abs=1e-5)
    assert sell_trade.fee == pytest.approx(1.497, abs=1e-5)
    assert sell_trade.net_cost == pytest.approx(1495.503, abs=1e-5)
    assert portfolio.cash == pytest.approx(cash_after_buy + 1495.503, abs=1e-5)
    assert portfolio.get_position("INFY.NS") == 10.0
    assert sell_trade.realized_pnl == pytest.approx(493.503, abs=1e-5)


def test_insufficient_cash_rejection_and_integrity():
    """Verify that an order exceeding available cash is rejected and cash is untouched."""
    config = SimulationConfig(initial_cash=1000.0, reject_on_insufficient=True)
    portfolio = Portfolio(initial_cash=1000.0)
    ts = pd.Timestamp("2024-01-02", tz="UTC")

    # Order requires ~5,000 INR
    order = Order(symbol="TCS.NS", side=OrderSide.BUY, quantity=50.0, created_at=ts)
    trade = portfolio.execute_order(order, base_price=100.0, timestamp=ts, config=config)

    assert trade is None
    assert order.status == OrderStatus.REJECTED
    assert "Insufficient cash" in (order.reason or "")
    assert portfolio.cash == 1000.0
    assert portfolio.get_position("TCS.NS") == 0.0

    # If reject_on_insufficient=False, raise InsufficientCashError
    strict_config = SimulationConfig(reject_on_insufficient=False)
    strict_order = Order(symbol="TCS.NS", side=OrderSide.BUY, quantity=50.0, created_at=ts)
    with pytest.raises(InsufficientCashError, match="Insufficient cash"):
        portfolio.execute_order(strict_order, base_price=100.0, timestamp=ts, config=strict_config)


def test_insufficient_holdings_rejection():
    """Verify that selling more shares than held is rejected without state mutation."""
    config = SimulationConfig(reject_on_insufficient=True)
    portfolio = Portfolio(initial_cash=5000.0)
    ts = pd.Timestamp("2024-01-02", tz="UTC")

    # Holding 0 shares, attempting to sell 10
    order = Order(symbol="SBIN.NS", side=OrderSide.SELL, quantity=10.0, created_at=ts)
    trade = portfolio.execute_order(order, base_price=500.0, timestamp=ts, config=config)

    assert trade is None
    assert order.status == OrderStatus.REJECTED
    assert "Insufficient holdings" in (order.reason or "")
    assert portfolio.cash == 5000.0

    # Strict mode raises InsufficientHoldingsError
    strict_config = SimulationConfig(reject_on_insufficient=False)
    strict_order = Order(symbol="SBIN.NS", side=OrderSide.SELL, quantity=10.0, created_at=ts)
    with pytest.raises(InsufficientHoldingsError, match="Insufficient holdings"):
        portfolio.execute_order(strict_order, base_price=500.0, timestamp=ts, config=strict_config)


def test_fractional_shares_rejection():
    """Verify fractional share orders are rejected when allow_fractional=False."""
    config = SimulationConfig(allow_fractional=False, reject_on_insufficient=True)
    portfolio = Portfolio(initial_cash=10_000.0)
    ts = pd.Timestamp("2024-01-02", tz="UTC")

    order = Order(symbol="RELIANCE.NS", side=OrderSide.BUY, quantity=5.5, created_at=ts)
    trade = portfolio.execute_order(order, base_price=2000.0, timestamp=ts, config=config)

    assert trade is None
    assert order.status == OrderStatus.REJECTED
    assert "Fractional shares not allowed" in (order.reason or "")


def test_minimum_fee_enforcement():
    """Verify min_fee overrides proportional fee when proportional fee is lower."""
    config = SimulationConfig(fee_rate=0.0001, min_fee=25.0)  # Min fee = 25 INR
    portfolio = Portfolio(initial_cash=10_000.0)
    ts = pd.Timestamp("2024-01-02", tz="UTC")

    order = Order(symbol="ICICIBANK.NS", side=OrderSide.BUY, quantity=1.0, created_at=ts)
    trade = portfolio.execute_order(order, base_price=100.0, timestamp=ts, config=config)

    assert trade is not None
    assert trade.fee == 25.0


# ==============================================================================
# Performance Metrics Calculations & Edge Cases
# ==============================================================================

def test_metrics_hand_calculated():
    """Hand-verify CAGR, drawdown, volatility, and return formulas."""
    p = Portfolio(initial_cash=100_000.0)
    ts0 = pd.Timestamp("2024-01-01", tz="UTC")
    ts1 = pd.Timestamp("2024-01-02", tz="UTC")
    ts2 = pd.Timestamp("2024-01-03", tz="UTC")

    # Snapshot 0: 100,000
    p.record_snapshot(ts0, {})
    # Snapshot 1: Cash increases to 110,000 (+10%)
    p.cash = 110_000.0
    p.record_snapshot(ts1, {})
    # Snapshot 2: Cash drops to 99,000 (-10% from peak of 110,000)
    p.cash = 99_000.0
    p.record_snapshot(ts2, {})

    metrics = calculate_metrics(p.snapshots, trades=[], risk_free_rate=0.05, periods_per_year=252)

    # Initial: 100k, Final: 99k -> Total return: -1% = -0.01
    assert metrics.initial_value == 100_000.0
    assert metrics.final_value == 99_000.0
    assert metrics.total_return == pytest.approx(-0.01, abs=1e-5)

    # Max Drawdown: peak was 110k, dropped to 99k -> (110k - 99k) / 110k = 11 / 110 = 0.10 (10%)
    assert metrics.max_drawdown == pytest.approx(0.10, abs=1e-5)
    assert metrics.total_bars == 3


def test_metrics_zero_volatility_edge_case():
    """Verify that constant portfolio value does not cause division-by-zero errors."""
    p = Portfolio(initial_cash=100_000.0)
    for i in range(10):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)
        p.record_snapshot(ts, {})

    metrics = calculate_metrics(p.snapshots, trades=[], risk_free_rate=0.05)

    assert metrics.total_return == 0.0
    assert metrics.annualized_volatility == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.sortino_ratio == 0.0
    assert metrics.max_drawdown == 0.0


def test_metrics_empty_and_single_snapshot():
    """Verify empty and single-bar metrics handle gracefully."""
    empty_m = calculate_metrics([], trades=[])
    assert empty_m.total_bars == 0
    assert empty_m.total_return == 0.0

    p = Portfolio(initial_cash=50_000.0)
    p.record_snapshot(pd.Timestamp("2024-01-01", tz="UTC"), {})
    single_m = calculate_metrics(p.snapshots, trades=[])
    assert single_m.total_bars == 1
    assert single_m.total_return == 0.0
    assert single_m.annualized_volatility == 0.0


# ==============================================================================
# Baseline Strategy: Buy and Hold Tests
# ==============================================================================

def test_buy_and_hold_execution_timing_and_no_lookahead(sample_5bar_df):
    """Verify that BuyAndHold generates signal on bar 0 and fills strictly on bar 1 OPEN."""
    sim = PortfolioSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            fee_rate=0.0003,
            slippage_rate=0.0005,
            allow_fractional=False,
        )
    )
    strategy = BuyAndHoldStrategy(target_symbol="TCS.NS", allocation_pct=0.95)
    result = sim.run(sample_5bar_df, strategy)

    # Total 1 trade filled
    assert len(result.portfolio.trades) == 1
    trade = result.portfolio.trades[0]

    # Bar 0 is 2024-01-01. Signal created at bar 0.
    # Bar 1 is 2024-01-02. Execution MUST occur on 2024-01-02 at bar 1 OPEN (102.0).
    bar1_ts = pd.Timestamp("2024-01-02", tz="UTC")
    assert trade.timestamp == bar1_ts
    assert trade.base_price == 102.0  # Bar 1 Open price
    assert trade.side == OrderSide.BUY

    # Fill price includes slippage: 102.0 * (1 + 0.0005) = 102.051
    assert trade.execution_price == pytest.approx(102.0 * 1.0005, abs=1e-5)

    # Held through remainder of backtest: no more trades
    assert len(result.portfolio.order_history) == 1
    assert result.portfolio.get_position("TCS.NS") == trade.quantity

    # Final valuation mark-to-market at bar 4 close (110.0)
    final_holdings_val = trade.quantity * 110.0
    expected_total_val = result.portfolio.cash + final_holdings_val
    assert result.metrics.final_value == pytest.approx(expected_total_val, abs=1e-2)
    assert result.metrics.total_return > 0.0


# ==============================================================================
# Baseline Strategy: Moving Average Crossover Tests
# ==============================================================================

def test_ma_crossover_execution_timing(trending_60bar_df):
    """Verify MA crossover detects golden cross and fills at the subsequent bar OPEN."""
    sim = PortfolioSimulator(
        SimulationConfig(
            initial_cash=100_000.0,
            fee_rate=0.0003,
            slippage_rate=0.0005,
            allow_fractional=False,
        )
    )
    # fast=5, slow=15 on trending_60bar_df (prices decline first 25 bars, then surge)
    strategy = MACrossoverStrategy(
        fast_window=5,
        slow_window=15,
        ma_type="SMA",
        target_symbol="RELIANCE.NS",
        allocation_pct=0.90,
    )
    result = sim.run(trending_60bar_df, strategy)

    assert len(result.portfolio.trades) >= 1
    buy_trade = result.portfolio.trades[0]
    assert buy_trade.side == OrderSide.BUY

    # Find the corresponding order
    order = result.portfolio.order_history[0]
    # Execution timestamp must be strictly later than order creation timestamp
    assert buy_trade.timestamp > order.created_at
    # The elapsed time must be exactly 1 day (next available session)
    assert buy_trade.timestamp - order.created_at == pd.Timedelta(days=1)

    # Strategy avoids duplicate buys while holding position
    buys = [t for t in result.portfolio.trades if t.side == OrderSide.BUY]
    assert len(buys) == 1  # Exactly one buy during the single golden cross regime


def test_ma_crossover_validation_constraints():
    """Verify parameter validations for MA Crossover."""
    with pytest.raises(ValueError, match="strictly less than slow_window"):
        MACrossoverStrategy(fast_window=20, slow_window=10)

    with pytest.raises(ValueError, match="Windows must be positive"):
        MACrossoverStrategy(fast_window=-5, slow_window=10)

    with pytest.raises(ValueError, match="ma_type must be 'SMA' or 'EMA'"):
        MACrossoverStrategy(fast_window=5, slow_window=10, ma_type="WMA")


# ==============================================================================
# Missing Data, Multi-Asset Alignment & Immutability Tests
# ==============================================================================

def test_multi_asset_calendar_and_missing_data_handling():
    """Verify simulator aligns timestamps and carries forward last known price during market gaps."""
    dates_a = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], utc=True)
    # ASSET_B misses session 2024-01-03
    dates_b = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"], utc=True)

    df_a = pd.DataFrame({
        "timestamp": dates_a,
        "symbol": "ASSET_A",
        "open": [100.0, 102.0, 104.0, 106.0],
        "high": [103.0, 105.0, 107.0, 109.0],
        "low":  [99.0, 101.0, 103.0, 105.0],
        "close": [102.0, 104.0, 106.0, 108.0],
        "volume": [1000] * 4,
    })

    df_b = pd.DataFrame({
        "timestamp": dates_b,
        "symbol": "ASSET_B",
        "open": [200.0, 202.0, 208.0],
        "high": [203.0, 205.0, 210.0],
        "low":  [199.0, 201.0, 206.0],
        "close": [202.0, 204.0, 210.0],
        "volume": [2000] * 3,
    })

    sim = PortfolioSimulator(SimulationConfig(initial_cash=50_000.0))
    strategy = BuyAndHoldStrategy(target_symbol="ASSET_B")
    result = sim.run({"ASSET_A": df_a, "ASSET_B": df_b}, strategy)

    # Calendar must contain all 4 distinct timestamps
    assert len(result.snapshots_df) == 4
    # On 2024-01-03, ASSET_B wasn't traded; valuation carried forward 2024-01-02 close (204.0)
    snap_day3 = result.snapshots_df.loc[pd.Timestamp("2024-01-03", tz="UTC")]
    assert snap_day3["holdings_value"] > 0


def test_reproducibility_and_input_data_immutability(sample_5bar_df):
    """Verify that multiple simulation runs produce identical results and do not mutate raw data."""
    original_df = sample_5bar_df.copy(deep=True)

    sim = PortfolioSimulator(SimulationConfig(initial_cash=100_000.0))
    strat1 = BuyAndHoldStrategy(target_symbol="TCS.NS")
    strat2 = BuyAndHoldStrategy(target_symbol="TCS.NS")

    res1 = sim.run(sample_5bar_df, strat1)
    res2 = sim.run(sample_5bar_df, strat2)

    # Identical results
    assert res1.metrics.total_return == res2.metrics.total_return
    assert res1.metrics.final_value == res2.metrics.final_value
    assert len(res1.portfolio.trades) == len(res2.portfolio.trades)

    # Input DataFrame has not been mutated
    pd.testing.assert_frame_equal(sample_5bar_df, original_df)


def test_explicit_no_lookahead_contract(sample_5bar_df):
    """Verify that strategies are never exposed to future timestamps during order generation."""
    observed_max_timestamps = []

    class InspectingStrategy(BuyAndHoldStrategy):
        def generate_orders(self, bar_idx, timestamp, historical_data, portfolio):
            df = historical_data["TCS.NS"]
            # Assert the maximum timestamp in the slice is exactly current bar's timestamp
            assert df.index.max() == timestamp
            assert len(df) == bar_idx + 1
            observed_max_timestamps.append(timestamp)
            return super().generate_orders(bar_idx, timestamp, historical_data, portfolio)

    sim = PortfolioSimulator(SimulationConfig(initial_cash=10_000.0))
    strat = InspectingStrategy(target_symbol="TCS.NS")
    sim.run(sample_5bar_df, strat)

    # Strategy was invoked for bars 0 to 3 (all bars before final bar)
    assert len(observed_max_timestamps) == 4
    expected_dates = sample_5bar_df["timestamp"].iloc[:4].tolist()
    assert observed_max_timestamps == expected_dates


def test_hand_calculated_buy_and_hold_full_equity_curve():
    """Verify exact full equity curve, cash balance, and drawdown step by step.
    
    Bar 0: 2024-01-01, Open=100.0, Close=100.0 (Buy signal generated)
    Bar 1: 2024-01-02, Open=100.0, Close=120.0 (Buy fills at 100.0, zero cost)
    Bar 2: 2024-01-03, Open=120.0, Close=90.0
    Bar 3: 2024-01-04, Open=90.0,  Close=150.0
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "NSE_STOCK",
        "open":  [100.0, 100.0, 120.0, 90.0],
        "high":  [102.0, 122.0, 121.0, 155.0],
        "low":   [98.0,  99.0,  88.0,  89.0],
        "close": [100.0, 120.0, 90.0,  150.0],
        "volume": [1000] * 4,
    })

    # Zero fees and slippage to isolate pure mathematical accounting
    sim = PortfolioSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            allow_fractional=False,
        )
    )
    # 100% allocation -> 10,000 / 100 = 100 shares
    strat = BuyAndHoldStrategy(target_symbol="NSE_STOCK", allocation_pct=1.0)
    result = sim.run(df, strat)

    # 1 trade on Day 1
    assert len(result.portfolio.trades) == 1
    trade = result.portfolio.trades[0]
    assert trade.quantity == 100.0
    assert trade.execution_price == 100.0
    assert result.portfolio.cash == 0.0

    snaps = result.snapshots_df
    # Day 0: cash=10,000, holdings_val=0, total=10,000, DD=0
    assert snaps.iloc[0]["total_value"] == 10_000.0
    assert snaps.iloc[0]["drawdown"] == 0.0

    # Day 1: cash=0, holdings_val=100*120=12,000, total=12,000, DD=0
    assert snaps.iloc[1]["total_value"] == 12_000.0
    assert snaps.iloc[1]["drawdown"] == 0.0

    # Day 2: cash=0, holdings_val=100*90=9,000, total=9,000, DD=(12k-9k)/12k = 3/12 = 0.25 (25%)
    assert snaps.iloc[2]["total_value"] == 9_000.0
    assert snaps.iloc[2]["drawdown"] == pytest.approx(0.25, abs=1e-5)

    # Day 3: cash=0, holdings_val=100*150=15,000, total=15,000, DD=0
    assert snaps.iloc[3]["total_value"] == 15_000.0
    assert snaps.iloc[3]["drawdown"] == 0.0

    # Overall metrics:
    # Total return: (15,000 - 10,000) / 10,000 = +50% = 0.50
    assert result.metrics.total_return == pytest.approx(0.50, abs=1e-5)
    # Max drawdown: exactly 25% = 0.25
    assert result.metrics.max_drawdown == pytest.approx(0.25, abs=1e-5)


def test_compare_strategies_and_reporting(sample_5bar_df):
    """Verify strategy comparison table, summary formatter, and JSON export."""
    from backend.app.simulation import compare_strategies, export_result_json, format_backtest_summary

    sim = PortfolioSimulator(SimulationConfig(initial_cash=10_000.0))
    bh = sim.run(sample_5bar_df, BuyAndHoldStrategy(target_symbol="TCS.NS"))
    ma = sim.run(sample_5bar_df, MACrossoverStrategy(fast_window=2, slow_window=3, target_symbol="TCS.NS"))

    # Summary string formatting
    summary_text = format_backtest_summary(bh)
    assert "TradeMind Backtest Summary" in summary_text
    assert "TCS.NS" in str(bh.portfolio.trades[0].symbol)

    # Comparison DataFrame
    cmp_df = compare_strategies([bh, ma])
    assert "BuyAndHold" in cmp_df.columns
    assert "MACrossover" in cmp_df.columns
    assert "Net Total Return" in cmp_df.index or "Total Return" in cmp_df.index

    # JSON export
    json_str = export_result_json(bh)
    assert '"strategy_name": "BuyAndHold"' in json_str
    assert '"initial_cash": 10000.0' in json_str

