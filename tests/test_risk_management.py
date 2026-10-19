"""TradeMind Phase 7 Unit & Integration Tests: Risk Management Engine.

Deterministic, offline test suite verifying:
- RiskConfig validation, boundary defaults, and conservative settings.
- Pre-trade order and action evaluation (BUY permitted vs rejected vs constrained).
- Exposure limits (single-asset position limit and multi-asset portfolio limit).
- Insufficient cash protection without negative cash or phantom fees.
- Peak-to-trough drawdown calculation and threshold breach handling.
- Daily session loss breach and cooldown countdown/release mechanics.
- Invalid price, non-finite portfolio NAV, and corrupt valuation safety lockout.
- Auditable intervention logging (status, reason, telemetry, to_dict).
- Integration with Phase 4 PortfolioSimulator (zero phantom fees, clean audit).
- Integration with Phase 5 Gymnasium TradingEnvironment (action override to HOLD).
- Backward compatibility when risk engine is disabled or None.
- Repeated-run determinism and reproducible results.
"""
from __future__ import annotations

import math
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from backend.app.risk import (
    InterventionReason,
    InterventionStatus,
    RiskConfig,
    RiskError,
    RiskIntervention,
    RiskLimitBreachError,
    RiskManager,
)
from backend.app.rl.actions import TradingAction
from backend.app.rl.config import EnvConfig
from backend.app.rl.environment import TradingEnvironment
from backend.app.simulation.engine import PortfolioSimulator
from backend.app.simulation.models import (
    Order,
    OrderSide,
    OrderStatus,
    SimulationConfig,
)
from backend.app.simulation.portfolio import Portfolio
from backend.app.simulation.strategies.base import BaseStrategy


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def default_risk_config() -> RiskConfig:
    """Standard conservative risk management configuration."""
    return RiskConfig(
        max_position_exposure=0.50,
        max_portfolio_exposure=0.80,
        max_drawdown_limit=0.15,
        max_daily_loss_limit=0.04,
        cooldown_bars=3,
        log_interventions=True,
        enabled=True,
    )


@pytest.fixture
def risk_manager(default_risk_config: RiskConfig) -> RiskManager:
    """Pre-initialized RiskManager instance."""
    rm = RiskManager(default_risk_config)
    rm.reset(initial_cash=100_000.0)
    return rm


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """65-bar synthetic price series for environment and simulator integration."""
    dates = pd.date_range("2024-01-01", periods=65, freq="D", tz="UTC")
    # Steady rise for 40 bars, then sharp decline
    p_up = np.linspace(100.0, 150.0, 40)
    p_down = np.linspace(145.0, 90.0, 25)
    close = np.concatenate([p_up, p_down])
    open_ = np.roll(close, 1)
    open_[0] = 99.0
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    volume = np.full(65, 10_000, dtype=np.int64)

    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "INFY.NS",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================

def test_risk_config_defaults():
    """Verify conservative defaults suitable for simulation experiments."""
    cfg = RiskConfig()
    assert cfg.enabled is True
    assert 0.0 < cfg.max_position_exposure <= 1.0
    assert 0.0 < cfg.max_portfolio_exposure <= 1.0
    assert 0.0 < cfg.max_drawdown_limit <= 1.0
    assert 0.0 < cfg.max_daily_loss_limit <= 1.0
    assert cfg.cooldown_bars >= 0
    assert cfg.log_interventions is True


def test_risk_config_validation_invalid_bounds():
    """Verify ValueError is raised on non-positive or out-of-range parameters."""
    with pytest.raises(ValueError, match="max_position_exposure"):
        RiskConfig(max_position_exposure=0.0)
    with pytest.raises(ValueError, match="max_position_exposure"):
        RiskConfig(max_position_exposure=2.5)

    with pytest.raises(ValueError, match="max_portfolio_exposure"):
        RiskConfig(max_portfolio_exposure=-0.1)

    with pytest.raises(ValueError, match="max_drawdown_limit"):
        RiskConfig(max_drawdown_limit=1.1)

    with pytest.raises(ValueError, match="max_daily_loss_limit"):
        RiskConfig(max_daily_loss_limit=0.0)

    with pytest.raises(ValueError, match="cooldown_bars"):
        RiskConfig(cooldown_bars=-1)


# ==============================================================================
# 2. Risk Manager State & Reset Tests
# ==============================================================================

def test_risk_manager_initialization_and_reset(risk_manager: RiskManager):
    """Verify clean state tracking and deterministic reset behavior."""
    risk_manager.current_drawdown = 0.20
    risk_manager.is_restricted = True
    risk_manager.cooldown_remaining = 5
    risk_manager._log_intervention(
        timestamp="2024-01-01",
        step=1,
        symbol="INFY.NS",
        proposed_action="BUY",
        status=InterventionStatus.REJECTED,
        reason=InterventionReason.MAX_DRAWDOWN_EXCEEDED,
        details="Test intervention",
        resulting_action="HOLD",
    )
    assert len(risk_manager.interventions) == 1

    risk_manager.reset(initial_cash=200_000.0)
    assert risk_manager.peak_nav == 200_000.0
    assert risk_manager.prev_nav == 200_000.0
    assert risk_manager.current_drawdown == 0.0
    assert risk_manager.current_daily_loss == 0.0
    assert risk_manager.cooldown_remaining == 0
    assert risk_manager.is_restricted is False
    assert len(risk_manager.interventions) == 0


# ==============================================================================
# 3. Pre-Trade Order & Price Validation Tests
# ==============================================================================

def test_evaluate_order_valid_buy(risk_manager: RiskManager):
    """Verify compliant BUY order within exposure limits is approved."""
    portfolio = Portfolio(initial_cash=100_000.0)
    order = Order(
        symbol="TCS.NS",
        side=OrderSide.BUY,
        quantity=100.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = risk_manager.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=1,
    )

    assert approved_order is not None
    assert approved_order.quantity == 100.0
    assert intervention is None
    assert len(risk_manager.interventions) == 0


@pytest.mark.parametrize("bad_price", [0.0, -10.0, float("nan"), float("inf")])
def test_evaluate_order_invalid_price(risk_manager: RiskManager, bad_price: float):
    """Verify invalid or non-finite prices immediately reject orders without fees."""
    portfolio = Portfolio(initial_cash=100_000.0)
    order = Order(
        symbol="TCS.NS",
        side=OrderSide.BUY,
        quantity=50.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = risk_manager.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=bad_price,
        timestamp="2024-01-03",
        step=2,
    )

    assert approved_order is None
    assert intervention is not None
    assert intervention.status == InterventionStatus.REJECTED
    assert intervention.reason == InterventionReason.INVALID_PRICE
    assert intervention.resulting_action == "REJECT"


def test_evaluate_order_insufficient_cash_reject(risk_manager: RiskManager):
    """Verify order requesting capital exceeding available cash is rejected."""
    portfolio = Portfolio(initial_cash=5_000.0)
    # Propose buying 100 shares @ 100.0 = 10,000 INR > 5,000 cash
    order = Order(
        symbol="TCS.NS",
        side=OrderSide.BUY,
        quantity=100.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = risk_manager.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=3,
    )

    assert approved_order is None
    assert intervention is not None
    assert intervention.status == InterventionStatus.REJECTED
    assert intervention.reason == InterventionReason.INSUFFICIENT_CASH


def test_evaluate_order_insufficient_cash_boundary():
    """Verify order requesting capital exceeding available cash is strictly rejected without phantom fill."""
    cfg = RiskConfig(max_position_exposure=1.0)
    rm = RiskManager(cfg)
    rm.reset(initial_cash=5_000.0)

    portfolio = Portfolio(initial_cash=5_000.0)
    # Propose buying 100 shares @ 100.0 = 10,000 INR > 5,000 available cash
    order = Order(
        symbol="TCS.NS",
        side=OrderSide.BUY,
        quantity=100.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = rm.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=3,
    )

    assert approved_order is None
    assert intervention is not None
    assert intervention.status == InterventionStatus.REJECTED
    assert intervention.reason == InterventionReason.INSUFFICIENT_CASH


# ==============================================================================
# 4. Position & Exposure Limits Tests
# ==============================================================================

def test_evaluate_order_single_asset_position_limit_exceeded(risk_manager: RiskManager):
    """Verify BUY order pushing position above max_position_exposure (50%) is constrained/rejected."""
    portfolio = Portfolio(initial_cash=100_000.0)
    # Propose buying 800 shares @ 100.0 = 80,000 INR (80% of NAV > 50% limit)
    order = Order(
        symbol="TCS.NS",
        side=OrderSide.BUY,
        quantity=800.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = risk_manager.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=4,
    )

    assert intervention is not None
    assert intervention.reason == InterventionReason.POSITION_EXPOSURE_EXCEEDED
    if approved_order is not None:
        assert approved_order.quantity <= 500.0


def test_evaluate_order_portfolio_exposure_limit_exceeded():
    """Verify BUY order pushing total portfolio exposure above limit is constrained."""
    cfg = RiskConfig(
        max_position_exposure=1.0,
        max_portfolio_exposure=0.70,
    )
    rm = RiskManager(cfg)
    rm.reset(initial_cash=100_000.0)

    portfolio = Portfolio(initial_cash=100_000.0)
    # Already holding 50,000 INR of ASSET_A (50% portfolio exposure)
    portfolio.holdings["ASSET_A"] = 500.0
    portfolio.cash = 50_000.0

    # Propose buying 40,000 INR of ASSET_B (would result in 90% exposure > 70% limit)
    order = Order(
        symbol="ASSET_B",
        side=OrderSide.BUY,
        quantity=400.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = rm.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=5,
    )

    assert intervention is not None
    assert intervention.reason == InterventionReason.PORTFOLIO_EXPOSURE_EXCEEDED
    if approved_order is not None:
        # Max additional exposure allowed is 20% of 100,000 = 20,000 INR = ~199 shares
        assert approved_order.quantity <= 200.0


def test_sell_order_always_permitted_when_restricted(risk_manager: RiskManager):
    """Verify SELL order is strictly allowed even when risk-restricted or in cooldown."""
    risk_manager.is_restricted = True
    risk_manager.cooldown_remaining = 5

    portfolio = Portfolio(initial_cash=20_000.0)
    portfolio.holdings["TCS.NS"] = 100.0

    order = Order(
        symbol="TCS.NS",
        side=OrderSide.SELL,
        quantity=100.0,
        created_at="2024-01-02",
    )
    approved_order, intervention = risk_manager.evaluate_proposed_order(
        order=order,
        portfolio=portfolio,
        execution_price=100.0,
        timestamp="2024-01-03",
        step=6,
    )

    assert approved_order is not None
    assert approved_order.quantity == 100.0
    assert intervention is None


# ==============================================================================
# 5. Drawdown, Daily Loss & Cooldown Release Tests
# ==============================================================================

def test_drawdown_calculation_and_breach(risk_manager: RiskManager):
    """Verify point-in-time drawdown formula (peak - current) / peak and breach trigger."""
    # Peak starts at 100,000. Update NAV to 120,000 (new peak)
    risk_manager.update_portfolio_valuation("2024-01-02", 120_000.0, step=1)
    assert risk_manager.peak_nav == 120_000.0
    assert risk_manager.current_drawdown == 0.0
    assert risk_manager.is_restricted is False

    # NAV drops to 100,000: drawdown = (120,000 - 100,000) / 120,000 = 16.67% > 15% limit
    risk_manager.update_portfolio_valuation("2024-01-03", 100_000.0, step=2)
    expected_dd = (120_000.0 - 100_000.0) / 120_000.0
    assert math.isclose(risk_manager.current_drawdown, expected_dd, rel_tol=1e-5)
    assert risk_manager.is_restricted is True
    assert risk_manager.cooldown_remaining == risk_manager.config.cooldown_bars

    # Check intervention recorded
    assert len(risk_manager.interventions) == 1
    assert risk_manager.interventions[0].reason == InterventionReason.MAX_DRAWDOWN_EXCEEDED
    assert risk_manager.interventions[0].status == InterventionStatus.MODIFIED

    # In restricted state, new BUY is blocked
    portfolio = Portfolio(initial_cash=100_000.0)
    order = Order(symbol="TCS.NS", side=OrderSide.BUY, quantity=10.0, created_at="2024-01-03")
    approved, interv = risk_manager.evaluate_proposed_order(order, portfolio, 100.0, "2024-01-04", step=3)
    assert approved is None
    assert interv is not None
    assert interv.status == InterventionStatus.REJECTED
    assert interv.reason == InterventionReason.MAX_DRAWDOWN_EXCEEDED


def test_daily_loss_calculation_and_breach(risk_manager: RiskManager):
    """Verify daily session loss formula (prev - current) / prev and breach trigger."""
    risk_manager.update_portfolio_valuation("2024-01-02", 100_000.0, step=1)

    # Next day drops to 94,000: loss = 6% > 4% daily limit
    risk_manager.update_portfolio_valuation("2024-01-03", 94_000.0, step=2)
    assert math.isclose(risk_manager.current_daily_loss, 0.06, rel_tol=1e-5)
    assert risk_manager.is_restricted is True
    assert risk_manager.cooldown_remaining == 3
    assert risk_manager.interventions[0].reason == InterventionReason.DAILY_LOSS_EXCEEDED


def test_cooldown_decay_and_release(risk_manager: RiskManager):
    """Verify cooldown counts down on non-breach bars and releases restriction deterministically."""
    # Trigger 5% daily loss breach (limit 4%)
    risk_manager.update_portfolio_valuation("2024-01-02", 100_000.0, step=1)
    risk_manager.update_portfolio_valuation("2024-01-03", 95_000.0, step=2)
    assert risk_manager.is_restricted is True
    assert risk_manager.cooldown_remaining == 3

    # Bar 1 of cooldown: NAV flat at 95,000 (0% loss, DD = 5% < 15%)
    risk_manager.update_portfolio_valuation("2024-01-04", 95_000.0, step=3)
    assert risk_manager.cooldown_remaining == 2
    assert risk_manager.is_restricted is True

    # Bar 2 of cooldown: NAV recovers to 96,000
    risk_manager.update_portfolio_valuation("2024-01-05", 96_000.0, step=4)
    assert risk_manager.cooldown_remaining == 1
    assert risk_manager.is_restricted is True

    # Bar 3 of cooldown: NAV flat at 96,000 -> Cooldown expires!
    risk_manager.update_portfolio_valuation("2024-01-06", 96_000.0, step=5)
    assert risk_manager.cooldown_remaining == 0
    assert risk_manager.is_restricted is False


def test_invalid_and_corrupt_portfolio_valuation(risk_manager: RiskManager):
    """Verify NaN, Inf, or negative NAV safely locks down trading without unhandled exceptions."""
    risk_manager.update_portfolio_valuation("2024-01-02", float("nan"), step=1)
    assert risk_manager.is_restricted is True
    assert len(risk_manager.interventions) == 1
    assert risk_manager.interventions[0].reason == InterventionReason.INVALID_PORTFOLIO_VALUE

    risk_manager.reset(initial_cash=100_000.0)
    risk_manager.update_portfolio_valuation("2024-01-02", -500.0, step=1)
    assert risk_manager.is_restricted is True
    assert risk_manager.interventions[0].reason == InterventionReason.INVALID_PORTFOLIO_VALUE


# ==============================================================================
# 6. Auditable Intervention Logging Tests
# ==============================================================================

def test_intervention_telemetry_and_serialization(risk_manager: RiskManager):
    """Verify RiskIntervention fields, to_dict serialization, and audit integrity."""
    risk_manager.update_portfolio_valuation("2024-01-02", 100_000.0, step=1)
    risk_manager.update_portfolio_valuation("2024-01-03", 80_000.0, step=2)  # 20% DD breach

    intervention = risk_manager.interventions[0]
    data = intervention.to_dict()

    assert data["timestamp"] == "2024-01-03"
    assert data["step"] == 2
    assert data["symbol"] == "PORTFOLIO"
    assert data["status"] == "MODIFIED"
    assert data["reason"] == "MAX_DRAWDOWN_EXCEEDED"
    assert data["resulting_action"] == "RESTRICTED"
    assert math.isclose(data["current_drawdown"], 0.20, rel_tol=1e-5)
    assert data["portfolio_value"] == 80_000.0


# ==============================================================================
# 7. Phase 4 Portfolio Simulator Integration Tests
# ==============================================================================

class AlwaysBuyStrategy(BaseStrategy):
    """Synthetic test strategy requesting BUY on every single bar."""
    def generate_orders(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Order]:
        orders = []
        for sym in self.symbols:
            orders.append(
                Order(
                    symbol=sym,
                    side=OrderSide.BUY,
                    quantity=100.0,
                    created_at=timestamp,
                )
            )
        return orders

    def reset(self) -> None:
        """Reset internal strategy state."""
        pass


def test_simulator_with_risk_manager_blocks_phantom_trades(sample_ohlcv_df: pd.DataFrame):
    """Verify PortfolioSimulator with RiskManager blocks orders during severe drawdown without phantom fees."""
    risk_cfg = RiskConfig(
        max_position_exposure=1.0,
        max_drawdown_limit=0.10,  # 10% drawdown limit
        cooldown_bars=10,
        enabled=True,
    )
    rm = RiskManager(risk_cfg)

    sim_cfg = SimulationConfig(
        initial_cash=100_000.0,
        fee_rate=0.001,
        slippage_rate=0.001,
    )
    simulator = PortfolioSimulator(config=sim_cfg, risk_manager=rm)
    result = simulator.run(sample_ohlcv_df, AlwaysBuyStrategy())

    # Strategy attempts to BUY on every bar
    # In the second half of the data, prices drop sharply from 150 to 90 (40% crash)
    # The risk manager should trigger a drawdown breach and block further BUY orders
    assert len(result.risk_interventions) > 0

    # Ensure blocked orders did NOT create phantom trades or charge duplicate fees
    rejected_orders = [o for o in result.portfolio.order_history if o.status == OrderStatus.REJECTED]
    assert len(rejected_orders) > 0

    # For every rejected order, ensure it has NO corresponding executed trade
    executed_order_ids = {t.order_id for t in result.portfolio.trades}
    for rej_order in rejected_orders:
        assert rej_order.order_id not in executed_order_ids


def test_simulator_without_risk_manager_backward_compatibility(sample_ohlcv_df: pd.DataFrame):
    """Verify simulation result with risk_manager=None or enabled=False matches baseline exactly."""
    sim_cfg = SimulationConfig(initial_cash=100_000.0)

    # 1. Simulator with risk_manager=None
    sim_none = PortfolioSimulator(config=sim_cfg, risk_manager=None)
    res_none = sim_none.run(sample_ohlcv_df, AlwaysBuyStrategy())

    # 2. Simulator with risk_manager disabled
    rm_disabled = RiskManager(RiskConfig(enabled=False))
    sim_disabled = PortfolioSimulator(config=sim_cfg, risk_manager=rm_disabled)
    res_disabled = sim_disabled.run(sample_ohlcv_df, AlwaysBuyStrategy())

    # Check numerical and ledger equivalence
    assert len(res_none.portfolio.trades) == len(res_disabled.portfolio.trades)
    assert math.isclose(
        res_none.portfolio.get_total_value({"INFY.NS": sample_ohlcv_df["close"].iloc[-1]}),
        res_disabled.portfolio.get_total_value({"INFY.NS": sample_ohlcv_df["close"].iloc[-1]}),
    )
    assert len(res_disabled.risk_interventions) == 0


# ==============================================================================
# 8. Phase 5 Gymnasium Trading Environment Integration Tests
# ==============================================================================

def test_rl_environment_action_override_and_info(sample_ohlcv_df: pd.DataFrame):
    """Verify TradingEnvironment overrides BUY to HOLD when restricted and populates info dict."""
    risk_cfg = RiskConfig(
        max_drawdown_limit=0.05,  # Tight 5% limit
        cooldown_bars=5,
        enabled=True,
    )
    rm = RiskManager(risk_cfg)

    env_cfg = EnvConfig(
        initial_cash=100_000.0,
        warmup_bars=10,
        allocation_pct=0.95,
    )
    env = TradingEnvironment(sample_ohlcv_df, config=env_cfg, risk_manager=rm)
    obs, info = env.reset()

    # Artificially trigger risk restriction on the manager
    rm.is_restricted = True
    rm.cooldown_remaining = 5

    # Agent selects BUY (Action 1)
    obs, reward, terminated, truncated, info = env.step(int(TradingAction.BUY))

    # Verification:
    # Proposed action was BUY (1), but effective action must be HOLD (0)
    assert info["action"] == int(TradingAction.BUY)
    assert info["effective_action"] == int(TradingAction.HOLD)
    assert info["trade_executed"] is False
    assert info["risk_restricted"] is True
    assert info["risk_intervention"] is not None
    assert info["risk_intervention"]["status"] == "REJECTED"
    assert info["risk_intervention"]["resulting_action"] == "HOLD"
    assert len(env.portfolio.trades) == 0


def test_rl_environment_backward_compatibility(sample_ohlcv_df: pd.DataFrame):
    """Verify TradingEnvironment operates identically when risk_manager is None."""
    env_cfg = EnvConfig(initial_cash=100_000.0, warmup_bars=10)
    env = TradingEnvironment(sample_ohlcv_df, config=env_cfg, risk_manager=None)
    obs, info = env.reset()

    assert env.risk_manager is None
    assert info["risk_restricted"] is False
    assert info["risk_intervention"] is None

    # Step BUY action
    obs, reward, terminated, truncated, info = env.step(int(TradingAction.BUY))
    assert info["action"] == int(TradingAction.BUY)
    assert info["effective_action"] == int(TradingAction.BUY)
    assert info["trade_executed"] is True
    assert len(env.portfolio.trades) == 1


# ==============================================================================
# 9. Reproducibility & Determinism Tests
# ==============================================================================

def test_repeated_run_determinism(sample_ohlcv_df: pd.DataFrame):
    """Verify identical configurations and inputs produce bit-for-bit identical risk decisions."""
    def run_sim():
        rm = RiskManager(RiskConfig(max_drawdown_limit=0.12, cooldown_bars=4))
        sim = PortfolioSimulator(config=SimulationConfig(initial_cash=100_000.0), risk_manager=rm)
        return sim.run(sample_ohlcv_df, AlwaysBuyStrategy())

    res_1 = run_sim()
    res_2 = run_sim()

    assert len(res_1.risk_interventions) == len(res_2.risk_interventions)
    for i1, i2 in zip(res_1.risk_interventions, res_2.risk_interventions):
        assert i1.to_dict() == i2.to_dict()

    assert len(res_1.portfolio.trades) == len(res_2.portfolio.trades)
    for t1, t2 in zip(res_1.portfolio.trades, res_2.portfolio.trades):
        assert t1.execution_price == t2.execution_price
        assert t1.fee == t2.fee
