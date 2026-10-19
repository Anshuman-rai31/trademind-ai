"""TradeMind Phase 5 Gymnasium Trading Environment.

Institutional single-asset reinforcement learning trading environment adhering
to the Gymnasium standard interface. Reuses Phase 4 Portfolio ledger, execution
timing, slippage, and brokerage fee models with zero lookahead bias.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ..features.engineer import FeatureConfig, apply_all_features
from ..simulation.models import (
    Order,
    OrderSide,
    OrderStatus,
    SimulationConfig,
    Trade,
)
from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..risk.models import RiskIntervention
from ..simulation.portfolio import Portfolio
from .actions import TradingAction
from .config import EnvConfig

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Gymnasium-compliant quantitative spot equity trading environment.
    
    Action Space:
        Discrete(3):
            0 = HOLD (maintain current position; execute zero trades)
            1 = BUY  (allocate available cash buffer into long position)
            2 = SELL (liquidate open position to 100% cash)

    Observation Space:
        Box(low=-10.0, high=10.0, shape=(N,), dtype=np.float32)
        Combines normalized Phase 3 technical features with internal portfolio
        state metrics (cash ratio, position ratio, unrealized PnL, drawdown).

    Timing Sequence (Strict Zero-Lookahead):
        1. Agent observes market features and portfolio state at bar t close.
        2. Agent chooses action a_t in {0, 1, 2}.
        3. Action executes at bar t+1 OPEN with realistic slippage and transaction fee.
        4. Portfolio is marked to market at bar t+1 CLOSE.
        5. Reward is the log return of total portfolio equity: ln(NAV_{t+1} / NAV_t).
        6. Environment advances to bar t+1; next observation is returned.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        symbol: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        """Initialize TradingEnvironment.
        
        Args:
            df: OHLCV DataFrame with price data and optional feature indicators.
            config: Optional EnvConfig hyperparameters.
            symbol: Ticker symbol (inferred from df['symbol'] if available).
            start_idx: First bar index of episode (must be >= warmup_bars).
            end_idx: Final bar index of episode (defaults to len(df)).
        """
        super().__init__()
        self.config: EnvConfig = config or EnvConfig()
        self.sim_config: SimulationConfig = self.config.to_simulation_config()

        # Normalize and prepare data
        self.df, self.symbol = self._prepare_data(df, symbol)
        self.total_rows: int = len(self.df)

        if self.total_rows < self.config.warmup_bars + 3:
            raise ValueError(
                f"DataFrame has {self.total_rows} rows, but requires at least "
                f"{self.config.warmup_bars + 3} rows for warmup and execution."
            )

        # Set episode boundaries
        self.start_idx: int = (
            max(self.config.warmup_bars, start_idx)
            if start_idx is not None
            else self.config.warmup_bars
        )
        self.end_idx: int = min(self.total_rows, end_idx) if end_idx is not None else self.total_rows

        if self.start_idx >= self.end_idx - 1:
            raise ValueError(
                f"Invalid episode slice: start_idx={self.start_idx} must be strictly "
                f"less than end_idx - 1 ({self.end_idx - 1})"
            )

        # Configure observation feature column names
        self.feature_cols: List[str] = self._determine_feature_columns()

        # Portfolio state features: cash_ratio, holdings_ratio, position_flag, unrealized_return, drawdown
        self.portfolio_feature_count: int = 5 if self.config.include_portfolio_state else 0
        self.obs_dim: int = len(self.feature_cols) + self.portfolio_feature_count

        # Define Gymnasium Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=float(self.config.min_obs_bound),
            high=float(self.config.max_obs_bound),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Internal state variables
        self.portfolio: Portfolio = Portfolio(initial_cash=self.config.initial_cash)
        self.current_step: int = self.start_idx
        self._steps_taken: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._last_action: int = int(TradingAction.HOLD)
        self._last_effective_action: int = int(TradingAction.HOLD)
        self._last_trade: Optional[Trade] = None
        self._last_intervention: Optional[RiskIntervention] = None
        self.risk_interventions: List[RiskIntervention] = []

        # Risk management layer
        if risk_manager is not None:
            self.risk_manager: Optional[RiskManager] = risk_manager
        elif self.config.risk_config is not None:
            self.risk_manager = RiskManager(config=self.config.risk_config)
        else:
            self.risk_manager = None

    def _prepare_data(self, df: pd.DataFrame, symbol: Optional[str]) -> Tuple[pd.DataFrame, str]:
        """Validate input columns, ensure Phase 3 features, and set symbol."""
        clean = df.copy()
        clean.columns = [c.lower() for c in clean.columns]

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in clean.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required OHLCV columns: {missing}")

        inferred_symbol = symbol
        if not inferred_symbol and "symbol" in clean.columns:
            inferred_symbol = str(clean["symbol"].iloc[0])
        inferred_symbol = inferred_symbol or "ASSET"

        if "timestamp" not in clean.columns and isinstance(clean.index, pd.DatetimeIndex):
            clean = clean.reset_index()
            clean.rename(columns={clean.columns[0]: "timestamp"}, inplace=True)
        elif "timestamp" not in clean.columns:
            # Generate deterministic synthetic timestamps if absent
            clean["timestamp"] = pd.date_range("2024-01-01", periods=len(clean), freq="D", tz="UTC")

        # Automatically compute Phase 3 features if not already calculated
        if "simple_return" not in clean.columns or "rsi_14" not in clean.columns:
            feat_cfg = FeatureConfig(
                sma_windows=[5, 10, 20, 50],
                ema_windows=[9, 21],
                rsi_period=14,
                bb_window=20,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                atr_period=14,
            )
            clean = apply_all_features(clean, config=feat_cfg)

        return clean.reset_index(drop=True), inferred_symbol

    def _determine_feature_columns(self) -> List[str]:
        """Select normalized, finite market feature columns for observation."""
        if self.config.feature_columns is not None:
            missing = [c for c in self.config.feature_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"Specified feature columns not found in DataFrame: {missing}")
            return list(self.config.feature_columns)

        # Standard canonical normalized Phase 3 features
        candidate_cols = [
            "simple_return",
            "log_return",
            "rsi_14",
            "bb_pct_20_2.0",
            "macd_histogram",
            "atr_14",
            "close_zscore_20",
        ]
        available = [c for c in candidate_cols if c in self.df.columns]
        if not available:
            # Fallback to simple returns if specific indicators are absent
            if "simple_return" in self.df.columns:
                available = ["simple_return"]
            else:
                available = ["close"]
        return available

    def _get_market_features(self, step: int) -> np.ndarray:
        """Extract and normalize market observation features at bar step."""
        row = self.df.iloc[step]
        values: List[float] = []

        close_val = float(row["close"])
        safe_close = close_val if close_val > 0 else 1.0

        for col in self.feature_cols:
            val = float(row[col]) if col in row and not pd.isna(row[col]) else 0.0

            # Apply domain-appropriate feature normalization
            if col == "rsi_14":
                # Normalize RSI from [0, 100] to [-0.5, 0.5]
                val = (val / 100.0) - 0.5
            elif col == "macd_histogram":
                # Normalize MACD histogram by close price
                val = (val / safe_close) * 100.0
            elif col == "atr_14":
                # Normalize ATR by close price (volatility percentage)
                val = (val / safe_close) * 100.0
            elif col == "close":
                val = 0.0  # Raw nominal price is zeroed out to prevent scale bias

            values.append(val)

        return np.array(values, dtype=np.float32)

    def _get_portfolio_features(self, current_close: float) -> np.ndarray:
        """Construct normalized portfolio state features."""
        if not self.config.include_portfolio_state:
            return np.empty((0,), dtype=np.float32)

        nav = self.portfolio.get_total_value({self.symbol: current_close})
        safe_nav = nav if nav > 0 else self.config.initial_cash

        cash = self.portfolio.cash
        holdings = self.portfolio.get_position(self.symbol)
        holdings_val = holdings * current_close

        cash_ratio = np.clip(cash / safe_nav, 0.0, 1.0)
        holdings_ratio = np.clip(holdings_val / safe_nav, 0.0, 1.0)
        position_flag = 1.0 if holdings > 0 else 0.0

        # Unrealized return on open position
        cost_basis = self.portfolio.cost_basis.get(self.symbol, 0.0)
        if holdings > 0 and cost_basis > 0:
            unrealized_return = (current_close - cost_basis) / cost_basis
        else:
            unrealized_return = 0.0

        # Current portfolio drawdown
        drawdown = 0.0
        if self.portfolio.peak_value > 0:
            drawdown = max(0.0, (self.portfolio.peak_value - nav) / self.portfolio.peak_value)

        features = [
            float(cash_ratio),
            float(holdings_ratio),
            float(position_flag),
            float(unrealized_return),
            float(drawdown),
        ]
        return np.array(features, dtype=np.float32)

    def _get_observation(self, step: int) -> np.ndarray:
        """Construct complete observation vector at step t."""
        current_close = float(self.df.iloc[step]["close"])
        mkt_feats = self._get_market_features(step)
        port_feats = self._get_portfolio_features(current_close)

        obs = np.concatenate([mkt_feats, port_feats])
        # Clean any accidental NaNs/Infs and clip strictly within observation space
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.config.max_obs_bound, neginf=self.config.min_obs_bound)
        return np.clip(obs, self.config.min_obs_bound, self.config.max_obs_bound).astype(np.float32)

    def _get_info(self, action: Optional[int], trade: Optional[Trade]) -> Dict[str, Any]:
        """Construct comprehensive evaluation and debug info dictionary."""
        current_ts = self.df.iloc[self.current_step]["timestamp"]
        current_close = float(self.df.iloc[self.current_step]["close"])
        nav = self.portfolio.get_total_value({self.symbol: current_close})
        holdings = self.portfolio.get_position(self.symbol)

        return {
            "step": self.current_step,
            "steps_taken": self._steps_taken,
            "timestamp": str(current_ts),
            "symbol": self.symbol,
            "portfolio_value": float(nav),
            "cash": float(self.portfolio.cash),
            "holdings": float(holdings),
            "holdings_value": float(holdings * current_close),
            "close_price": float(current_close),
            "action": int(action) if action is not None else int(TradingAction.HOLD),
            "action_name": TradingAction(action).name if action is not None else "HOLD",
            "effective_action": int(self._last_effective_action),
            "effective_action_name": TradingAction(self._last_effective_action).name,
            "risk_intervention": self._last_intervention.to_dict() if self._last_intervention else None,
            "risk_restricted": self.risk_manager.is_restricted if self.risk_manager else False,
            "risk_cooldown_remaining": self.risk_manager.cooldown_remaining if self.risk_manager else 0,
            "trade_executed": trade is not None,
            "trade_fee": float(trade.fee) if trade else 0.0,
            "trade_slippage": float(trade.slippage) if trade else 0.0,
            "total_trades": len(self.portfolio.trades),
            "peak_value": float(self.portfolio.peak_value),
            "drawdown": float(self.portfolio.snapshots[-1].drawdown) if self.portfolio.snapshots else 0.0,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of the episode.
        
        Args:
            seed: Optional integer seed for reproducible initialization.
            options: Optional configuration overrides (e.g. {'start_idx': 60}).
            
        Returns:
            Tuple of (initial_observation, initial_info_dict).
        """
        super().reset(seed=seed)

        # Handle options override
        if options and "start_idx" in options:
            self.current_step = max(self.config.warmup_bars, int(options["start_idx"]))
        else:
            self.current_step = self.start_idx

        self._steps_taken = 0
        self._terminated = False
        self._truncated = False
        self._last_action = int(TradingAction.HOLD)
        self._last_effective_action = int(TradingAction.HOLD)
        self._last_trade = None
        self._last_intervention = None
        self.risk_interventions = []

        # Reset RiskManager if configured
        if self.risk_manager is not None:
            self.risk_manager.reset(initial_cash=self.config.initial_cash)

        # Re-initialize clean portfolio ledger
        self.portfolio = Portfolio(initial_cash=self.config.initial_cash)
        self.sim_config = self.config.to_simulation_config()

        # Record initial mark-to-market snapshot at start bar close
        init_ts = self.df.iloc[self.current_step]["timestamp"]
        init_close = float(self.df.iloc[self.current_step]["close"])
        self.portfolio.record_snapshot(timestamp=init_ts, current_prices={self.symbol: init_close})

        # Initialize RiskManager tracking with starting NAV
        if self.risk_manager is not None and self.risk_manager.config.enabled:
            self.risk_manager.update_portfolio_valuation(
                timestamp=init_ts,
                current_nav=self.config.initial_cash,
                step=self.current_step,
            )

        obs = self._get_observation(self.current_step)
        info = self._get_info(action=None, trade=None)
        return obs, info

    def step(
        self,
        action: Union[int, np.integer, TradingAction],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step following strict no-lookahead timing.
        
        Sequence:
            1. Validate action.
            2. Execute trade at bar t+1 OPEN price (with slippage and transaction fee).
            3. Mark portfolio to market at bar t+1 CLOSE price.
            4. Compute log return reward: ln(NAV_{t+1} / NAV_t).
            5. Check termination / truncation conditions.
            6. Return observation at bar t+1.
            
        Args:
            action: 0 (HOLD), 1 (BUY), or 2 (SELL).
            
        Returns:
            5-tuple: (observation, reward, terminated, truncated, info).
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "Environment episode has finished (terminated or truncated). "
                "You must call reset() before calling step() again."
            )

        act = int(action)
        if not self.action_space.contains(act):
            raise ValueError(f"Action {act} is outside valid action space {self.action_space}")

        self._last_action = act
        t = self.current_step
        next_step = t + 1

        # Check if next execution bar exists
        if next_step >= self.end_idx:
            self._terminated = True
            obs = self._get_observation(t)
            info = self._get_info(action=act, trade=None)
            return obs, 0.0, True, False, info

        next_bar = self.df.iloc[next_step]
        next_open = float(next_bar[self.config.execution_price_col])
        next_close = float(next_bar[self.config.valuation_price_col])
        next_ts = next_bar["timestamp"]
        curr_close = float(self.df.iloc[t][self.config.valuation_price_col])

        # ----------------------------------------------------------------------
        # Pre-trade Risk Management Check
        # ----------------------------------------------------------------------
        effective_act = act
        intervention = None
        if self.risk_manager is not None and self.risk_manager.config.enabled:
            effective_act, intervention = self.risk_manager.evaluate_proposed_action(
                action=act,
                symbol=self.symbol,
                portfolio=self.portfolio,
                current_price=curr_close,
                next_open=next_open,
                step=t,
                timestamp=next_ts,
            )
            if intervention is not None:
                self.risk_interventions.append(intervention)

        self._last_intervention = intervention
        self._last_effective_action = effective_act

        trade: Optional[Trade] = None

        # ----------------------------------------------------------------------
        # Trade Execution at Bar t+1 Open (Realistic Slippage + Fees)
        # ----------------------------------------------------------------------
        if effective_act == TradingAction.BUY:
            current_holdings = self.portfolio.get_position(self.symbol)
            # Only enter if not already long (prevents unnecessary churn)
            if current_holdings <= 0 and next_open > 0:
                est_fill_price = next_open * (1.0 + self.config.slippage_rate)
                fee_factor = 1.0 + self.config.fee_rate
                cash_to_spend = self.portfolio.cash * self.config.allocation_pct
                target_shares = cash_to_spend / (est_fill_price * fee_factor)

                if not self.config.allow_fractional:
                    target_shares = float(int(target_shares))

                if target_shares >= self.config.min_trade_size:
                    order = Order(
                        symbol=self.symbol,
                        side=OrderSide.BUY,
                        quantity=target_shares,
                        created_at=self.df.iloc[t]["timestamp"],
                    )
                    trade = self.portfolio.execute_order(
                        order=order,
                        base_price=next_open,
                        timestamp=next_ts,
                        config=self.sim_config,
                    )

        elif effective_act == TradingAction.SELL:
            current_holdings = self.portfolio.get_position(self.symbol)
            # Only exit if currently holding shares
            if current_holdings > 0 and next_open > 0:
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=current_holdings,
                    created_at=self.df.iloc[t]["timestamp"],
                )
                trade = self.portfolio.execute_order(
                    order=order,
                    base_price=next_open,
                    timestamp=next_ts,
                    config=self.sim_config,
                )

        self._last_trade = trade

        # ----------------------------------------------------------------------
        # Mark to Market at Bar t+1 Close
        # ----------------------------------------------------------------------
        prev_nav = self.portfolio.snapshots[-1].total_value
        snapshot = self.portfolio.record_snapshot(
            timestamp=next_ts,
            current_prices={self.symbol: next_close},
        )
        curr_nav = snapshot.total_value

        # Update Risk Manager valuation tracking and cooldown countdown
        if self.risk_manager is not None and self.risk_manager.config.enabled:
            self.risk_manager.update_portfolio_valuation(
                timestamp=next_ts,
                current_nav=curr_nav,
                step=next_step,
            )

        # ----------------------------------------------------------------------
        # Log Portfolio Return Reward Calculation
        # ----------------------------------------------------------------------
        reward: float = 0.0
        if curr_nav <= 0.0 or prev_nav <= 0.0:
            # Catastrophic capital depletion penalty
            reward = -10.0 * self.config.reward_scaling
            self._terminated = True
        else:
            # Standard institutional log return: ln(NAV_{t+1} / NAV_t)
            reward = float(np.log(curr_nav / prev_nav) * self.config.reward_scaling)

        # Advance step counters
        self.current_step = next_step
        self._steps_taken += 1

        # ----------------------------------------------------------------------
        # Termination & Truncation Handling
        # ----------------------------------------------------------------------
        # Natural dataset exhaustion: Cannot step further if next_step is the final bar
        if self.current_step >= self.end_idx - 1:
            self._terminated = True

        # External step truncation
        if self.config.max_steps is not None and self._steps_taken >= self.config.max_steps:
            self._truncated = True

        # Observation for the newly advanced bar t+1
        obs = self._get_observation(self.current_step)
        info = self._get_info(action=act, trade=trade)

        return obs, reward, self._terminated, self._truncated, info

    def render(self) -> None:
        """Render current portfolio summary to standard log."""
        if not self.portfolio.snapshots:
            print("TradingEnvironment: Empty portfolio state.")
            return

        latest = self.portfolio.snapshots[-1]
        print(
            f"Step {self.current_step} [{latest.timestamp}] | "
            f"NAV: INR {latest.total_value:,.2f} | "
            f"Cash: INR {latest.cash:,.2f} | "
            f"Shares: {self.portfolio.get_position(self.symbol):.0f} | "
            f"Daily Ret: {latest.daily_return * 100:+.2f}% | "
            f"DD: {latest.drawdown * 100:.2f}%"
        )

    def close(self) -> None:
        """Clean up environment resources."""
        pass
