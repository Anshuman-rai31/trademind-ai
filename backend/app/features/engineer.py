"""TradeMind Feature Engineering Core Module (Phase 3).

Implements rigorous, stateless feature engineering for OHLCV financial time-series.

Features Computed
-----------------
Returns:
    - simple_return        : (close[t] - close[t-1]) / close[t-1]
    - log_return           : log(close[t] / close[t-1])

Moving Averages (configurable windows, default SMA: [5,10,20,50] EMA: [9,21]):
    - sma_{w}              : Simple Moving Average over w bars
    - ema_{w}              : Exponential Moving Average over w bars

Relative Strength Index:
    - rsi_{period}         : RSI using Wilder's smoothed average (default period=14)

Bollinger Bands (configurable window and sigma):
    - bb_upper_{w}_{s}     : Upper band = SMA + s * std
    - bb_lower_{w}_{s}     : Lower band = SMA - s * std
    - bb_mid_{w}_{s}       : Middle band = SMA
    - bb_width_{w}_{s}     : Band width = (upper - lower) / middle
    - bb_pct_{w}_{s}       : %B = (close - lower) / (upper - lower), NaN if bands == 0

MACD (configurable fast/slow/signal):
    - macd_line            : EMA(fast) - EMA(slow)
    - macd_signal          : EMA(macd_line, signal)
    - macd_histogram       : macd_line - macd_signal

Average True Range:
    - atr_{period}         : Wilder's smoothed ATR (default period=14)

On-Balance Volume:
    - obv                  : Cumulative directional volume

Normalization (rolling z-score, no global leakage):
    - {col}_zscore_{w}     : (col - rolling_mean) / rolling_std, default w=20

NaN Handling:
    Leading NaN rows from warm-up windows are preserved as-is so downstream
    consumers can choose to drop or mask them. The minimum warm-up length is
    reported in FeatureConfig.warmup_bars.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    COL_TIMESTAMP,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configurable knobs for all Phase 3 feature computations.

    All window/period parameters are enforced to be positive integers.
    Defaults match commonly used quant-research conventions.
    """
    # Moving averages
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_windows: List[int] = field(default_factory=lambda: [9, 21])

    # RSI
    rsi_period: int = 14

    # Bollinger Bands
    bb_window: int = 20
    bb_num_std: float = 2.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR
    atr_period: int = 14

    # Rolling z-score normalization columns and window
    zscore_columns: List[str] = field(
        default_factory=lambda: ["close", "volume", "simple_return"]
    )
    zscore_window: int = 20

    def __post_init__(self) -> None:
        """Validate all parameters immediately on construction."""
        _validate_windows("sma_windows", self.sma_windows)
        _validate_windows("ema_windows", self.ema_windows)
        _validate_period("rsi_period", self.rsi_period)
        _validate_period("bb_window", self.bb_window)
        if self.bb_num_std <= 0:
            raise ValueError(f"bb_num_std must be positive, got {self.bb_num_std}.")
        _validate_period("macd_fast", self.macd_fast)
        _validate_period("macd_slow", self.macd_slow)
        if self.macd_fast >= self.macd_slow:
            raise ValueError(
                f"macd_fast ({self.macd_fast}) must be < macd_slow ({self.macd_slow})."
            )
        _validate_period("macd_signal", self.macd_signal)
        _validate_period("atr_period", self.atr_period)
        _validate_period("zscore_window", self.zscore_window)

    @property
    def warmup_bars(self) -> int:
        """Minimum number of leading bars required before all features are non-NaN."""
        candidates = (
            list(self.sma_windows)
            + list(self.ema_windows)
            + [self.rsi_period + 1]
            + [self.bb_window]
            + [self.macd_slow + self.macd_signal - 1]
            + [self.atr_period + 1]
            + [self.zscore_window]
        )
        return max(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_windows(name: str, windows: Sequence[int], allow_empty: bool = False) -> None:
    if not windows:
        if allow_empty:
            return  # Empty list = skip this type; valid for compute_moving_averages
        raise ValueError(f"{name} must be a non-empty list.")
    for w in windows:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"All {name} must be positive integers. Got: {w}.")


def _validate_period(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got: {value}.")


def _require_column(df: pd.DataFrame, col: str, fn: str) -> None:
    if col not in df.columns:
        raise KeyError(f"[{fn}] Required column '{col}' not found in DataFrame.")


def _require_columns(df: pd.DataFrame, cols: Sequence[str], fn: str) -> None:
    for c in cols:
        _require_column(df, c, fn)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Returns
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append simple and log daily return columns to a copy of *df*.

    Columns added:
        simple_return : (close[t] - close[t-1]) / close[t-1]  — fractional
        log_return    : ln(close[t] / close[t-1])             — additive

    The first row is NaN for both because there is no prior close.
    Prices of zero are guarded: log_return is NaN where close == 0.

    Args:
        df: DataFrame containing at least a 'close' column (float, positive).

    Returns:
        New DataFrame (copy of *df*) with 'simple_return' and 'log_return' appended.

    Raises:
        KeyError: If 'close' column is absent.
    """
    fn = "compute_returns"
    _require_column(df, COL_CLOSE, fn)

    out = df.copy()
    close = out[COL_CLOSE].astype(float)

    out["simple_return"] = close.pct_change()

    # Guard against zero or negative close (should have been caught by validator,
    # but a belt-and-suspenders safeguard here prevents log domain errors)
    safe_ratio = close / close.shift(1)
    safe_ratio = safe_ratio.where(safe_ratio > 0, other=np.nan)
    out["log_return"] = np.log(safe_ratio)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Moving Averages
# ─────────────────────────────────────────────────────────────────────────────

def compute_moving_averages(
    df: pd.DataFrame,
    sma_windows: Optional[List[int]] = None,
    ema_windows: Optional[List[int]] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append SMA and EMA columns to a copy of *df*.

    Column naming:
        sma_{w}  — Simple Moving Average over the last w bars (inclusive)
        ema_{w}  — Exponential Moving Average, span=w, min_periods=w

    The first (w-1) bars of each window column are NaN.

    Args:
        df: DataFrame with a 'close' column.
        sma_windows: List of integer window sizes for SMA.
        ema_windows: List of integer span sizes for EMA.
        config: FeatureConfig providing defaults when windows are None.

    Returns:
        New DataFrame with SMA and EMA columns appended.
    """
    fn = "compute_moving_averages"
    _require_column(df, COL_CLOSE, fn)

    cfg = config or FeatureConfig()
    sma_wins = sma_windows if sma_windows is not None else cfg.sma_windows
    ema_wins = ema_windows if ema_windows is not None else cfg.ema_windows

    _validate_windows("sma_windows", sma_wins, allow_empty=True)
    _validate_windows("ema_windows", ema_wins, allow_empty=True)

    out = df.copy()
    close = out[COL_CLOSE].astype(float)

    for w in sma_wins:
        out[f"sma_{w}"] = close.rolling(window=w, min_periods=w).mean()

    for w in ema_wins:
        out[f"ema_{w}"] = close.ewm(span=w, min_periods=w, adjust=False).mean()

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — RSI (Wilder's Smoothed Average)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(
    df: pd.DataFrame,
    period: Optional[int] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append Wilder's RSI to a copy of *df*.

    Algorithm:
        delta  = close.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        avg_gain, avg_loss = exponential smoothing with alpha=1/period (Wilder)
        RS     = avg_gain / avg_loss
        RSI    = 100 - 100 / (1 + RS)

    Edge cases:
        - avg_loss == 0 (pure up streak): RSI = 100.0
        - avg_gain == 0 (pure down streak): RSI = 0.0

    Column added:
        rsi_{period}

    Args:
        df: DataFrame with 'close' column.
        period: RSI lookback in bars (default cfg.rsi_period = 14).
        config: FeatureConfig.

    Returns:
        New DataFrame with RSI column appended.
    """
    fn = "compute_rsi"
    _require_column(df, COL_CLOSE, fn)

    cfg = config or FeatureConfig()
    p = period if period is not None else cfg.rsi_period
    _validate_period("rsi_period", p)

    out = df.copy()
    close = out[COL_CLOSE].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing: alpha = 1/period, equivalent to com = period - 1
    alpha = 1.0 / p
    avg_gain = gain.ewm(alpha=alpha, min_periods=p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=p, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Edge: avg_loss == 0 → full bullish streak
    rsi = rsi.where(avg_loss != 0.0, other=100.0)
    # Edge: avg_gain == 0 → full bearish streak
    rsi = rsi.where(avg_gain != 0.0, other=0.0)

    # NaN until warm-up is satisfied (first `period+1` rows)
    out[f"rsi_{p}"] = rsi

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def compute_bollinger_bands(
    df: pd.DataFrame,
    window: Optional[int] = None,
    num_std: Optional[float] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append Bollinger Band columns to a copy of *df*.

    Columns added (w = window, s = formatted num_std):
        bb_mid_{w}_{s}    : Rolling SMA (middle band)
        bb_upper_{w}_{s}  : SMA + num_std * rolling_std
        bb_lower_{w}_{s}  : SMA - num_std * rolling_std
        bb_width_{w}_{s}  : (upper - lower) / mid   — normalised width
        bb_pct_{w}_{s}    : %B = (close - lower) / (upper - lower), NaN if flat

    The suffix *s* is the std multiplier formatted to 1 decimal (e.g. '2.0').

    Args:
        df: DataFrame with 'close' column.
        window: Rolling window length (default cfg.bb_window = 20).
        num_std: Number of standard deviations (default cfg.bb_num_std = 2.0).
        config: FeatureConfig.

    Returns:
        New DataFrame with Bollinger Band columns appended.
    """
    fn = "compute_bollinger_bands"
    _require_column(df, COL_CLOSE, fn)

    cfg = config or FeatureConfig()
    w = window if window is not None else cfg.bb_window
    s = num_std if num_std is not None else cfg.bb_num_std

    _validate_period("bb_window", w)
    if s <= 0:
        raise ValueError(f"num_std must be positive, got {s}.")

    suffix = f"{w}_{s:.1f}"
    out = df.copy()
    close = out[COL_CLOSE].astype(float)

    rolling = close.rolling(window=w, min_periods=w)
    mid = rolling.mean()
    std = rolling.std(ddof=1)

    upper = mid + s * std
    lower = mid - s * std
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    band_range = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / band_range

    out[f"bb_mid_{suffix}"] = mid
    out[f"bb_upper_{suffix}"] = upper
    out[f"bb_lower_{suffix}"] = lower
    out[f"bb_width_{suffix}"] = bandwidth
    out[f"bb_pct_{suffix}"] = pct_b

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — MACD
# ─────────────────────────────────────────────────────────────────────────────

def compute_macd(
    df: pd.DataFrame,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append MACD line, signal, and histogram columns to a copy of *df*.

    MACD Line      = EMA(close, fast) - EMA(close, slow)
    Signal Line    = EMA(MACD Line, signal)
    Histogram      = MACD Line - Signal Line

    Warm-up: (slow - 1) + (signal - 1) bars before histogram is non-NaN.

    Columns added:
        macd_line      : Raw MACD
        macd_signal    : Signal EMA
        macd_histogram : Divergence

    Args:
        df: DataFrame with 'close' column.
        fast: Fast EMA span (default cfg.macd_fast = 12).
        slow: Slow EMA span (default cfg.macd_slow = 26).
        signal: Signal EMA span (default cfg.macd_signal = 9).
        config: FeatureConfig.

    Returns:
        New DataFrame with MACD columns appended.
    """
    fn = "compute_macd"
    _require_column(df, COL_CLOSE, fn)

    cfg = config or FeatureConfig()
    f = fast if fast is not None else cfg.macd_fast
    sl = slow if slow is not None else cfg.macd_slow
    sig = signal if signal is not None else cfg.macd_signal

    _validate_period("macd_fast", f)
    _validate_period("macd_slow", sl)
    _validate_period("macd_signal", sig)
    if f >= sl:
        raise ValueError(f"macd_fast ({f}) must be < macd_slow ({sl}).")

    out = df.copy()
    close = out[COL_CLOSE].astype(float)

    ema_fast = close.ewm(span=f, min_periods=f, adjust=False).mean()
    ema_slow = close.ewm(span=sl, min_periods=sl, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_sig = macd_line.ewm(span=sig, min_periods=sig, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    out["macd_line"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_histogram"] = macd_hist

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Average True Range (ATR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(
    df: pd.DataFrame,
    period: Optional[int] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append Wilder's ATR to a copy of *df*.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR        = Wilder's exponential smoothing of TR with alpha=1/period

    The first bar TR equals (high - low) because there is no prior close.

    Column added:
        atr_{period}

    Args:
        df: DataFrame with 'high', 'low', and 'close' columns.
        period: ATR smoothing period (default cfg.atr_period = 14).
        config: FeatureConfig.

    Returns:
        New DataFrame with ATR column appended.
    """
    fn = "compute_atr"
    _require_columns(df, [COL_HIGH, COL_LOW, COL_CLOSE], fn)

    cfg = config or FeatureConfig()
    p = period if period is not None else cfg.atr_period
    _validate_period("atr_period", p)

    out = df.copy()
    high = out[COL_HIGH].astype(float)
    low = out[COL_LOW].astype(float)
    close = out[COL_CLOSE].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # First bar: no prev_close → TR = high - low (already correct from max)
    atr = tr.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    out[f"atr_{p}"] = atr

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — On-Balance Volume (OBV)
# ─────────────────────────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Append On-Balance Volume to a copy of *df*.

    Rule:
        If close[t] > close[t-1]  : OBV[t] = OBV[t-1] + volume[t]
        If close[t] < close[t-1]  : OBV[t] = OBV[t-1] - volume[t]
        If close[t] == close[t-1] : OBV[t] = OBV[t-1]

    OBV starts at 0 for the first bar. No warm-up required; full history is valid.

    Column added:
        obv

    Args:
        df: DataFrame with 'close' and 'volume' columns.

    Returns:
        New DataFrame with 'obv' column appended.
    """
    fn = "compute_obv"
    _require_columns(df, [COL_CLOSE, COL_VOLUME], fn)

    out = df.copy()
    close = out[COL_CLOSE].astype(float).values
    volume = out[COL_VOLUME].astype(float).values
    n = len(close)

    obv = np.zeros(n, dtype=float)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    out["obv"] = obv
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8 — Rolling Z-Score Normalization
# ─────────────────────────────────────────────────────────────────────────────

def compute_normalization(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    window: Optional[int] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Append rolling z-score normalized versions of specified columns.

    Z-score normalisation prevents global distribution leakage into earlier bars
    by using only the past *window* observations to compute mean and std.

    z = (x - rolling_mean(x, window)) / rolling_std(x, window)

    NaN is returned where rolling_std == 0 (flat price over window) to avoid
    division by zero. Leading NaN for the first (window - 1) bars.

    Columns added (for each col in *columns*):
        {col}_zscore_{window}

    Args:
        df: DataFrame containing the columns to normalize.
        columns: List of column names. Defaults to cfg.zscore_columns.
        window: Rolling window length. Defaults to cfg.zscore_window (20).
        config: FeatureConfig.

    Returns:
        New DataFrame with z-score columns appended.

    Warns:
        RuntimeWarning if a requested column is not found (column is skipped).
    """
    fn = "compute_normalization"
    cfg = config or FeatureConfig()
    cols = columns if columns is not None else cfg.zscore_columns
    w = window if window is not None else cfg.zscore_window

    _validate_period("zscore_window", w)

    out = df.copy()
    for col in cols:
        if col not in out.columns:
            logger.warning(
                "[%s] Column '%s' not found; skipping z-score normalization.", fn, col
            )
            continue
        series = out[col].astype(float)
        roll = series.rolling(window=w, min_periods=w)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=1).replace(0.0, np.nan)
        out[f"{col}_zscore_{w}"] = (series - roll_mean) / roll_std

    return out


# ─────────────────────────────────────────────────────────────────────────────
# High-Level Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_all_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Apply the full Phase 3 feature engineering pipeline to *df*.

    Chains all individual feature functions in a deterministic, documented order:
        1. compute_returns          → simple_return, log_return
        2. compute_moving_averages  → sma_{w}, ema_{w}
        3. compute_rsi              → rsi_{period}
        4. compute_bollinger_bands  → bb_*
        5. compute_macd             → macd_line, macd_signal, macd_histogram
        6. compute_atr              → atr_{period}
        7. compute_obv              → obv
        8. compute_normalization    → {col}_zscore_{w}

    The output DataFrame preserves the original OHLCV columns intact and appends
    all feature columns. NaN rows from warm-up periods are NOT dropped.

    Args:
        df: Validated OHLCV DataFrame from MarketDataPipeline.
        config: FeatureConfig instance (uses defaults if None).

    Returns:
        Augmented DataFrame with all feature columns added.

    Raises:
        KeyError: If a required base column is missing from *df*.
    """
    cfg = config or FeatureConfig()
    logger.debug("Applying feature engineering pipeline (warmup_bars=%d).", cfg.warmup_bars)

    out = compute_returns(df)
    out = compute_moving_averages(out, config=cfg)
    out = compute_rsi(out, config=cfg)
    out = compute_bollinger_bands(out, config=cfg)
    out = compute_macd(out, config=cfg)
    out = compute_atr(out, config=cfg)
    out = compute_obv(out)
    out = compute_normalization(out, config=cfg)

    logger.debug(
        "Feature pipeline complete. Columns: %d → %d. Rows: %d.",
        len(df.columns),
        len(out.columns),
        len(out),
    )
    return out
