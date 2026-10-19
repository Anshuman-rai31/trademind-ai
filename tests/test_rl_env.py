"""TradeMind Phase 5 Unit & Integration Tests: Gymnasium Trading Environment.

Comprehensive, deterministic test suite verifying:
- Gymnasium API compliance via gymnasium.utils.env_checker.check_env.
- Discrete action space (0=HOLD, 1=BUY, 2=SELL) and execution logic.
- Observation space bounds, finite values, and absence of NaNs.
- Strict no-lookahead timing: action at bar t executes at bar t+1 Open.
- Realistic transaction fee and slippage accounting.
- Log portfolio return reward calculation against closed-form examples.
- Cash constraints: no negative cash or negative holdings.
- Episode termination, truncation, and reset integrity.
- Seeded reproducibility and multi-asset isolation.
"""
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd
import pytest

from backend.app.rl import (
    EnvConfig,
    TradingAction,
    TradingEnvironment,
)


# ==============================================================================
# Deterministic Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_ohlcv_80():
    """80-bar deterministic price series with sufficient history for indicator warmup."""
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    
    # Predictable price curve: flat first 20 bars, gentle uptrend next 60 bars
    close = np.concatenate([
        np.full(20, 100.0),
        np.linspace(100.0, 160.0, 60),
    ])
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    volume = np.full(n, 50000.0)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "TCS.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def short_ohlcv_30():
    """30-bar dataset with warmup=5 for fast targeted tests."""
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100.0, 130.0, n)
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = np.full(n, 10000.0)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "RELIANCE.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ==============================================================================
# Gymnasium Compliance & Space Tests
# ==============================================================================

def test_gymnasium_check_env_compliance(synthetic_ohlcv_80):
    """Verify standard Gymnasium interface compliance using official check_env."""
    cfg = EnvConfig(warmup_bars=25)
    env = TradingEnvironment(synthetic_ohlcv_80, config=cfg)
    
    # check_env runs assertions on spaces, step returns, reset returns, and types
    check_env(env)


def test_observation_space_and_action_space_properties(synthetic_ohlcv_80):
    """Verify spaces, bounds, shapes, and dtypes."""
    cfg = EnvConfig(warmup_bars=20, min_obs_bound=-10.0, max_obs_bound=10.0)
    env = TradingEnvironment(synthetic_ohlcv_80, config=cfg)

    # Action Space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3

    # Observation Space
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert env.observation_space.shape == (env.obs_dim,)
    assert env.observation_space.low[0] == -10.0
    assert env.observation_space.high[0] == 10.0


def test_reset_output_and_seeding(synthetic_ohlcv_80):
    """Verify reset() returns valid initial observation and metadata dictionary."""
    cfg = EnvConfig(warmup_bars=20, initial_cash=75_000.0)
    env = TradingEnvironment(synthetic_ohlcv_80, config=cfg)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)

    # Info verification
    assert info["portfolio_value"] == 75_000.0
    assert info["cash"] == 75_000.0
    assert info["holdings"] == 0.0
    assert info["symbol"] == "TCS.NS"
    assert info["steps_taken"] == 0


# ==============================================================================
# Action Execution & Cash Constraint Tests
# ==============================================================================

def test_hold_action_does_not_trade(short_ohlcv_30):
    """Verify that HOLD (0) executes no orders and leaves cash untouched."""
    cfg = EnvConfig(warmup_bars=5, initial_cash=10_000.0)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(TradingAction.HOLD)

    assert info["action_name"] == "HOLD"
    assert not info["trade_executed"]
    assert info["holdings"] == 0.0
    assert info["cash"] == 10_000.0
    assert reward == 0.0  # NAV is unchanged when holding 100% cash


def test_buy_action_execution_and_slippage(short_ohlcv_30):
    """Verify BUY (1) fills at bar t+1 OPEN price with slippage and fee deducted."""
    cfg = EnvConfig(
        warmup_bars=5,
        initial_cash=10_000.0,
        fee_rate=0.001,       # 0.1%
        slippage_rate=0.002,  # 0.2%
        allocation_pct=0.90,
    )
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    t = env.current_step
    next_bar_open = float(short_ohlcv_30.iloc[t + 1]["open"])

    obs, reward, terminated, truncated, info = env.step(TradingAction.BUY)

    assert info["action_name"] == "BUY"
    assert info["trade_executed"]
    assert info["holdings"] > 0.0
    assert info["cash"] < 10_000.0
    assert env.portfolio.cash >= 0.0  # Strict cash non-negativity

    # Inspect executed trade in portfolio
    trade = env.portfolio.trades[0]
    expected_exec_price = next_bar_open * (1.0 + 0.002)
    assert trade.execution_price == pytest.approx(expected_exec_price, abs=1e-5)
    assert trade.fee == pytest.approx(trade.execution_price * trade.quantity * 0.001, abs=1e-5)


def test_sell_action_liquidates_position(short_ohlcv_30):
    """Verify SELL (2) liquidates open position and restores cash."""
    cfg = EnvConfig(warmup_bars=5, initial_cash=10_000.0, allocation_pct=0.90)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    # Step 1: BUY
    env.step(TradingAction.BUY)
    held_shares = env.portfolio.get_position("RELIANCE.NS")
    assert held_shares > 0

    # Step 2: SELL
    obs, reward, terminated, truncated, info = env.step(TradingAction.SELL)

    assert info["action_name"] == "SELL"
    assert info["trade_executed"]
    assert env.portfolio.get_position("RELIANCE.NS") == 0.0
    assert info["holdings"] == 0.0
    assert len(env.portfolio.trades) == 2


def test_sell_when_flat_is_safe_noop(short_ohlcv_30):
    """Verify that SELL when holding zero shares does not crash or execute a trade."""
    cfg = EnvConfig(warmup_bars=5, initial_cash=10_000.0)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    # In flat position, action SELL
    obs, reward, terminated, truncated, info = env.step(TradingAction.SELL)

    assert not info["trade_executed"]
    assert info["holdings"] == 0.0
    assert info["cash"] == 10_000.0
    assert len(env.portfolio.trades) == 0


def test_repeated_buy_is_safe_noop(short_ohlcv_30):
    """Verify that BUY while already in position is treated as no-op to avoid churn."""
    cfg = EnvConfig(warmup_bars=5, initial_cash=10_000.0, allocation_pct=0.90)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    # First BUY fills
    env.step(TradingAction.BUY)
    assert len(env.portfolio.trades) == 1
    shares_after_first = env.portfolio.get_position("RELIANCE.NS")

    # Second BUY while already holding
    obs, reward, terminated, truncated, info = env.step(TradingAction.BUY)

    assert not info["trade_executed"]
    assert len(env.portfolio.trades) == 1
    assert env.portfolio.get_position("RELIANCE.NS") == shares_after_first


# ==============================================================================
# Reward & Timing Tests
# ==============================================================================

def test_reward_hand_calculated():
    """Hand-verify reward calculation against closed-form log return."""
    # Construct exact 10-bar DataFrame with zero fee and zero slippage
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "NSE_STOCK",
        "open":  [100.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
        "high":  [101.0, 101.0, 111.0, 121.0, 131.0, 141.0, 151.0, 161.0, 171.0, 181.0],
        "low":   [99.0,  99.0,  109.0, 119.0, 129.0, 139.0, 149.0, 159.0, 169.0, 179.0],
        "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
        "volume": [1000] * 10,
    })

    cfg = EnvConfig(
        warmup_bars=0,
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        allocation_pct=1.0,
        reward_scaling=1.0,
    )
    env = TradingEnvironment(df, config=cfg, start_idx=0)
    env.reset()

    # Step 0: action BUY
    # At t=0 close, NAV_0 = 10,000
    # At t=1 open (100.0), BUY fills 100 shares at 100.0 = 10,000 cash spent.
    # At t=1 close (110.0), NAV_1 = 100 * 110.0 = 11,000.
    # Expected reward = ln(11,000 / 10,000) = ln(1.10)
    obs, reward, terminated, truncated, info = env.step(TradingAction.BUY)

    expected_reward = float(np.log(11_000.0 / 10_000.0))
    assert reward == pytest.approx(expected_reward, abs=1e-5)


def test_strict_no_lookahead_timing(short_ohlcv_30):
    """Verify action chosen at t executes at t+1 Open price, not t Open or t Close."""
    cfg = EnvConfig(warmup_bars=5, initial_cash=10_000.0, slippage_rate=0.0, fee_rate=0.0)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    t = env.current_step
    bar_t_close = float(short_ohlcv_30.iloc[t]["close"])
    bar_t_open = float(short_ohlcv_30.iloc[t]["open"])
    bar_next_open = float(short_ohlcv_30.iloc[t + 1]["open"])

    assert bar_next_open != bar_t_close
    assert bar_next_open != bar_t_open

    env.step(TradingAction.BUY)
    trade = env.portfolio.trades[0]

    # Fill price must be EXACTLY bar t+1 Open
    assert trade.base_price == bar_next_open
    assert trade.execution_price == bar_next_open


# ==============================================================================
# Episode Lifecycle & Boundary Tests
# ==============================================================================

def test_natural_episode_termination(short_ohlcv_30):
    """Verify episode terminates when dataset is exhausted."""
    cfg = EnvConfig(warmup_bars=5, max_steps=None)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    terminated = False
    step_count = 0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(TradingAction.HOLD)
        step_count += 1
        if step_count > 100:  # Safety guard
            break

    assert terminated
    assert env.current_step >= len(short_ohlcv_30) - 1

    # Stepping after termination raises RuntimeError
    with pytest.raises(RuntimeError, match="episode has finished"):
        env.step(TradingAction.HOLD)


def test_max_steps_truncation(short_ohlcv_30):
    """Verify episode truncates when steps_taken reaches max_steps."""
    cfg = EnvConfig(warmup_bars=5, max_steps=10)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    truncated = False
    step_count = 0
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(TradingAction.HOLD)
        step_count += 1

    assert truncated
    assert step_count == 10
    assert info["steps_taken"] == 10


def test_reset_after_completion_restores_state(short_ohlcv_30):
    """Verify that reset() cleanly resets all internal state after an episode finishes."""
    cfg = EnvConfig(warmup_bars=5, max_steps=5, initial_cash=50_000.0)
    env = TradingEnvironment(short_ohlcv_30, config=cfg)
    env.reset()

    # Run to truncation
    for _ in range(5):
        env.step(TradingAction.BUY)

    assert env._truncated

    # Reset
    obs, info = env.reset()
    assert not env._truncated
    assert not env._terminated
    assert env.portfolio.cash == 50_000.0
    assert env.portfolio.get_position("RELIANCE.NS") == 0.0
    assert len(env.portfolio.trades) == 0

    # Can step again cleanly
    obs2, reward, terminated, truncated, info2 = env.step(TradingAction.HOLD)
    assert not terminated


def test_reproducibility_with_seed(synthetic_ohlcv_80):
    """Verify that identical actions from seeded resets produce identical outputs."""
    cfg = EnvConfig(warmup_bars=20)
    env1 = TradingEnvironment(synthetic_ohlcv_80, config=cfg)
    env2 = TradingEnvironment(synthetic_ohlcv_80, config=cfg)

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)

    # Execute same action sequence
    actions = [TradingAction.BUY, TradingAction.HOLD, TradingAction.SELL, TradingAction.HOLD]
    for act in actions:
        o1, r1, term1, trunc1, inf1 = env1.step(act)
        o2, r2, term2, trunc2, inf2 = env2.step(act)
        np.testing.assert_array_almost_equal(o1, o2)
        assert r1 == pytest.approx(r2, abs=1e-6)
        assert inf1["portfolio_value"] == pytest.approx(inf2["portfolio_value"], abs=1e-2)


def test_gym_make_registered_environment(synthetic_ohlcv_80):
    """Verify that the environment can be instantiated via gym.make('TradeMind-v0')."""
    import backend.app.rl  # Trigger registration
    env = gym.make("TradeMind-v0", df=synthetic_ohlcv_80)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert not terminated
    env.close()


def test_multi_asset_isolation(synthetic_ohlcv_80, short_ohlcv_30):
    """Verify that environments running different tickers remain completely isolated."""
    cfg1 = EnvConfig(warmup_bars=10, initial_cash=100_000.0)
    cfg2 = EnvConfig(warmup_bars=5, initial_cash=50_000.0)

    env_tcs = TradingEnvironment(synthetic_ohlcv_80, config=cfg1, symbol="TCS.NS")
    env_rel = TradingEnvironment(short_ohlcv_30, config=cfg2, symbol="RELIANCE.NS")

    env_tcs.reset()
    env_rel.reset()

    # Buy on TCS only
    env_tcs.step(TradingAction.BUY)
    # Hold on RELIANCE
    env_rel.step(TradingAction.HOLD)

    assert env_tcs.portfolio.get_position("TCS.NS") > 0
    assert env_tcs.portfolio.get_position("RELIANCE.NS") == 0
    assert env_rel.portfolio.get_position("RELIANCE.NS") == 0
    assert env_rel.portfolio.get_position("TCS.NS") == 0
    assert env_rel.portfolio.cash == 50_000.0


def test_insufficient_cash_when_price_exceeds_capital():
    """Verify safe handling when single share price exceeds initial capital."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "EXPENSIVE_STOCK",
        "open":  [20000.0] * 10,  # 1 share costs 20,000 INR
        "high":  [20100.0] * 10,
        "low":   [19900.0] * 10,
        "close": [20000.0] * 10,
        "volume": [1000] * 10,
    })
    # Initial cash is only 1,000 INR -> cannot afford even 1 share
    cfg = EnvConfig(warmup_bars=1, initial_cash=1000.0, allow_fractional=False)
    env = TradingEnvironment(df, config=cfg)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(TradingAction.BUY)

    assert not info["trade_executed"]
    assert env.portfolio.cash == 1000.0
    assert env.portfolio.get_position("EXPENSIVE_STOCK") == 0.0
    assert reward == 0.0

