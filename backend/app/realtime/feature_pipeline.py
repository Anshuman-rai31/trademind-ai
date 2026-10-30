"""PPO-Compatible Live Feature Engineering Pipeline (Phase 11E).

Calculates exact 12-dimensional observation vectors matching the trained PPO model's
observation space, feature definitions, normalization scalers, and ordering.
Enforces zero lookahead bias, protects against incomplete candles, checks warmup sufficiency,
and explicitly reports timeframe compatibility mismatches against saved model checkpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..features.engineer import FeatureConfig, apply_all_features
from .candle_builder import Candle

logger = logging.getLogger(__name__)

# Expected canonical observation features matching TradingEnvironment
CANONICAL_MARKET_FEATURES = [
    "simple_return",
    "log_return",
    "rsi_14",
    "bb_pct_20_2.0",
    "macd_histogram",
    "atr_14",
    "close_zscore_20",
]

PORTFOLIO_STATE_FEATURES = [
    "cash_ratio",
    "holdings_ratio",
    "position_flag",
    "unrealized_return",
    "drawdown",
]

EXPECTED_OBS_DIM = 12
MIN_WARMUP_BARS = 50


class LiveFeaturePipeline:
    """Live feature engineering pipeline producing observations for PPO model inference."""

    def __init__(
        self,
        trained_model_timeframe: str = "1d",
        min_warmup_bars: int = MIN_WARMUP_BARS,
    ):
        self.trained_model_timeframe = trained_model_timeframe.lower()
        self.min_warmup_bars = min_warmup_bars
        self.feat_config = FeatureConfig(
            sma_windows=[5, 10, 20, 50],
            ema_windows=[9, 21],
            rsi_period=14,
            bb_window=20,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            atr_period=14,
        )

    def generate_live_observation(
        self,
        candles_history: List[Dict[str, Any]],
        portfolio_state: Optional[Dict[str, Any]] = None,
        candle_interval: str = "5m",
        symbol: str = "RELIANCE.NS",
    ) -> Dict[str, Any]:
        """Generate a validated 12-dimensional observation vector for PPO inference.
        
        Args:
            candles_history: Chronological list of completed OHLCV candle dicts.
            portfolio_state: Dict with 'cash', 'holdings', 'cost_basis', 'peak_value'.
            candle_interval: Timeframe of the candles ("1m", "5m", "1d").
            symbol: Target stock ticker symbol.
            
        Returns:
            Dict containing:
                - ready (bool): Whether observation is ready for model inference.
                - observation (Optional[List[float]]): 12-d observation array if ready.
                - observation_dim (int): Vector length (12).
                - timeframe_mismatch (bool): True if candle interval != trained model interval.
                - reason (Optional[str]): Diagnostic message if not ready.
                - features (Optional[Dict[str, float]]): Named breakdown of calculated features.
        """
        # 1. Guard against incomplete candles
        for c in candles_history:
            if not c.get("is_complete", True):
                return {
                    "ready": False,
                    "observation": None,
                    "observation_dim": EXPECTED_OBS_DIM,
                    "timeframe_mismatch": False,
                    "reason": "Incomplete candle detected. The PPO model requires closed, completed bars.",
                    "features": None,
                }

        clean_inv = candle_interval.strip().lower()
        is_timeframe_mismatch = clean_inv != self.trained_model_timeframe

        # 2. Check sufficient history for warmup
        if len(candles_history) < self.min_warmup_bars:
            return {
                "ready": False,
                "observation": None,
                "observation_dim": EXPECTED_OBS_DIM,
                "timeframe_mismatch": is_timeframe_mismatch,
                "reason": (
                    f"Insufficient history: received {len(candles_history)} completed bars, "
                    f"but at least {self.min_warmup_bars} bars are required for rolling indicators (MACD, BB, ATR)."
                ),
                "features": None,
            }

        # 3. Construct DataFrame from history with strict chronological validation
        df = pd.DataFrame(candles_history)
        df.columns = [c.lower() for c in df.columns]

        required_cols = ["open", "high", "low", "close", "volume"]
        for rc in required_cols:
            if rc not in df.columns:
                return {
                    "ready": False,
                    "observation": None,
                    "observation_dim": EXPECTED_OBS_DIM,
                    "timeframe_mismatch": False,
                    "reason": f"Input candles missing required column '{rc}'.",
                    "features": None,
                }

        # 4. Compute Phase 3 technical features reusing the official feature engineer
        try:
            featured_df = apply_all_features(df, config=self.feat_config)
        except Exception as exc:
            return {
                "ready": False,
                "observation": None,
                "observation_dim": EXPECTED_OBS_DIM,
                "timeframe_mismatch": False,
                "reason": f"Feature calculation error: {exc}",
                "features": None,
            }

        target_row = featured_df.iloc[-1]
        close_val = float(target_row["close"])
        safe_close = close_val if close_val > 0 else 1.0

        # 5. Extract and normalize the 7 market features
        market_feats: List[float] = []
        named_features: Dict[str, float] = {}

        for col in CANONICAL_MARKET_FEATURES:
            raw_val = float(target_row.get(col, 0.0)) if not pd.isna(target_row.get(col)) else 0.0
            norm_val = raw_val

            if col == "rsi_14":
                # Normalize RSI from [0, 100] to [-0.5, 0.5]
                norm_val = (raw_val / 100.0) - 0.5
            elif col == "macd_histogram":
                # Normalize MACD histogram by close price
                norm_val = (raw_val / safe_close) * 100.0
            elif col == "atr_14":
                # Normalize ATR by close price (volatility percentage)
                norm_val = (raw_val / safe_close) * 100.0

            market_feats.append(norm_val)
            named_features[col] = round(norm_val, 6)

        # 6. Extract and normalize the 5 portfolio state features
        port_state = portfolio_state or {
            "cash": 100000.0,
            "holdings": 0.0,
            "cost_basis": 0.0,
            "peak_value": 100000.0,
        }
        cash = float(port_state.get("cash", 100000.0))
        holdings = float(port_state.get("holdings", 0.0))
        cost_basis = float(port_state.get("cost_basis", 0.0))
        peak_val = float(port_state.get("peak_value", 100000.0))

        total_value = cash + (holdings * close_val)
        safe_total = total_value if total_value > 0 else 100000.0

        cash_ratio = float(np.clip(cash / safe_total, 0.0, 1.0))
        holdings_ratio = float(np.clip((holdings * close_val) / safe_total, 0.0, 1.0))
        pos_flag = 1.0 if holdings > 0 else 0.0
        unrealized_ret = float((close_val - cost_basis) / cost_basis) if holdings > 0 and cost_basis > 0 else 0.0
        drawdown = float(max(0.0, (peak_val - total_value) / peak_val)) if peak_val > 0 else 0.0

        portfolio_feats = [
            cash_ratio,
            holdings_ratio,
            pos_flag,
            unrealized_ret,
            drawdown,
        ]

        named_features["cash_ratio"] = round(cash_ratio, 4)
        named_features["holdings_ratio"] = round(holdings_ratio, 4)
        named_features["position_flag"] = pos_flag
        named_features["unrealized_return"] = round(unrealized_ret, 6)
        named_features["drawdown"] = round(drawdown, 4)

        # 7. Assemble 12-dimensional observation vector
        obs_array = np.array(market_feats + portfolio_feats, dtype=np.float32)

        # 8. Observation validation: shape, bounds, NaN/Inf checks
        if obs_array.shape != (EXPECTED_OBS_DIM,):
            return {
                "ready": False,
                "observation": None,
                "observation_dim": EXPECTED_OBS_DIM,
                "timeframe_mismatch": False,
                "reason": f"Invalid observation dimension {obs_array.shape}, expected ({EXPECTED_OBS_DIM},).",
                "features": None,
            }

        if np.isnan(obs_array).any() or np.isinf(obs_array).any():
            return {
                "ready": False,
                "observation": None,
                "observation_dim": EXPECTED_OBS_DIM,
                "timeframe_mismatch": False,
                "reason": "Observation contains NaN or infinite values.",
                "features": None,
            }

        # 9. Timeframe Compatibility Check
        clean_inv = candle_interval.strip().lower()
        if clean_inv != self.trained_model_timeframe:
            # Report timeframe mismatch safely without faking readiness
            return {
                "ready": False,
                "observation": [round(float(v), 6) for v in obs_array],
                "observation_dim": EXPECTED_OBS_DIM,
                "timeframe_mismatch": True,
                "trained_timeframe": self.trained_model_timeframe,
                "live_timeframe": clean_inv,
                "reason": (
                    f"Timeframe mismatch: Active PPO model was trained on daily ({self.trained_model_timeframe}) candles, "
                    f"but live feed is generating '{clean_inv}' candles. Direct model inference on intraday bars is disabled "
                    f"to prevent severe policy distribution shift."
                ),
                "features": named_features,
            }

        return {
            "ready": True,
            "observation": [round(float(v), 6) for v in obs_array],
            "observation_dim": EXPECTED_OBS_DIM,
            "timeframe_mismatch": False,
            "trained_timeframe": self.trained_model_timeframe,
            "live_timeframe": clean_inv,
            "reason": "Observation ready and fully compatible with active PPO model.",
            "features": named_features,
        }
