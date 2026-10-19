"""TradeMind Phase 8 Market-Regime Detection Engine.

Deterministic, leakage-safe unsupervised market-regime classification using K-Means clustering.
Strictly fits all preprocessing transformations (imputer, scaler) and clustering models
on historical training partitions only, preventing temporal lookahead leakage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIMESTAMP, COL_VOLUME
from ..data.split import ChronologicalSplits, chronological_split
from .config import RegimeConfig
from .exceptions import (
    DataLeakageError,
    FeatureExtractionError,
    InsufficientDataError,
    ModelNotFittedError,
    RegimeDetectionError,
)
from .models import ClusterSummary, RegimeModelMetadata, RegimeOutput

logger = logging.getLogger(__name__)


def compute_scale_free_regime_features(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """Extract scale-free financial features for market-regime detection.
    
    Guarantees:
        1. Returns, volatility, and momentum are intrinsically scale-free percentages
           or bounded oscillators, enabling direct comparison across equities with
           different nominal prices (e.g. INR 100 vs INR 3,500).
        2. If multiple symbols are present, rolling features are computed strictly
           within each symbol group, preventing cross-asset sequence contamination.
        3. Operates without future-data lookahead bias.
        
    Args:
        df: Input DataFrame containing OHLCV price series.
        config: Optional RegimeConfig specifying window lengths.
        
    Returns:
        DataFrame copy with appended regime indicator columns.
    """
    cfg = config or RegimeConfig()
    clean = df.copy()

    # Standardize column casing
    clean.columns = [c.lower() for c in clean.columns]

    required_base = ["close"]
    missing = [c for c in required_base if c not in clean.columns]
    if missing:
        raise FeatureExtractionError(f"Input DataFrame missing required price columns: {missing}")

    has_symbol = "symbol" in clean.columns
    unique_symbols = clean["symbol"].unique() if has_symbol else ["ASSET"]

    # Process grouped by symbol if multiple stocks present
    if has_symbol and len(unique_symbols) > 1:
        processed_groups: List[pd.DataFrame] = []
        for sym, grp in clean.groupby("symbol", sort=False):
            grp_sorted = grp.sort_values(by="timestamp" if "timestamp" in grp.columns else grp.index[0])
            grp_features = _compute_single_asset_regime_features(grp_sorted, cfg)
            processed_groups.append(grp_features)
        result = pd.concat(processed_groups, axis=0).sort_index()
        return result
    else:
        return _compute_single_asset_regime_features(clean, cfg)


def _compute_single_asset_regime_features(
    df: pd.DataFrame,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Compute regime features for a single contiguous asset time series."""
    out = df.copy()

    # 1. Daily Simple Return
    if "simple_return" not in out.columns:
        out["simple_return"] = out["close"].pct_change()

    # 2. Daily Log Return
    if "log_return" not in out.columns:
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # 3. Rolling Return Volatility (Annualized / Scaled)
    vol_col = f"volatility_{config.volatility_window}"
    if vol_col not in out.columns:
        # Rolling standard deviation of daily simple returns
        out[vol_col] = out["simple_return"].rolling(
            window=config.volatility_window,
            min_periods=max(2, config.volatility_window // 2),
        ).std()

    # 4. Momentum Feature (Wilder's RSI or Return Momentum)
    # Check if rsi_14 already exists from Phase 3
    if "rsi_14" in out.columns:
        out["momentum_rsi"] = out["rsi_14"]
    elif "momentum_rsi" not in out.columns:
        out["momentum_rsi"] = _compute_rsi(out["close"], period=config.momentum_period)

    # Secondary momentum: multi-bar cumulative return
    mom_ret_col = f"momentum_return_{config.momentum_period}"
    if mom_ret_col not in out.columns:
        out[mom_ret_col] = out["close"].pct_change(periods=config.momentum_period)

    return out


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's smoothed Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle pure gain or loss edge cases
    rsi = rsi.fillna(50.0)
    return rsi


class MarketRegimeDetector:
    """Deterministic, leakage-safe quantitative market-regime detection engine.
    
    Guarantees:
        1. Fit Exclusivity: Imputer, scaler, and clustering algorithms are fit ONLY
           on historical training data.
        2. Zero Lookahead: Validation and test data are strictly transformed using
           parameters learned on the training split.
        3. Neutral Cluster Interpretation: Cluster IDs (0, 1, 2) are arbitrary integers;
           human-readable labels are systematically derived from empirical training profiles.
        4. Cross-Stock Normalization: Features are scale-free percentage returns,
           rolling return volatilities, and normalized momentum metrics.
    """

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        """Initialize MarketRegimeDetector with configurable parameters.
        
        Args:
            config: Optional RegimeConfig instance.
        """
        self.config: RegimeConfig = config or RegimeConfig()
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.is_fitted: bool = False
        self.cluster_summaries: List[ClusterSummary] = []
        self.metadata: Optional[RegimeModelMetadata] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> MarketRegimeDetector:
        """Fit preprocessing pipeline and K-Means clustering strictly on training data.
        
        Args:
            train_df: Chronological training partition DataFrame.
            timestamp_col: Name of timestamp column.
            
        Returns:
            Fitted MarketRegimeDetector instance.
            
        Raises:
            InsufficientDataError: If training observations are below minimum threshold.
            FeatureExtractionError: If configured features cannot be extracted.
        """
        # 1. Prepare features if missing
        features_df = compute_scale_free_regime_features(train_df, self.config)

        # 2. Extract feature matrix
        missing_feats = [c for c in self.config.feature_names if c not in features_df.columns]
        if missing_feats:
            raise FeatureExtractionError(
                f"Configured feature_names {missing_feats} not found in prepared training data."
            )

        X_raw = features_df[self.config.feature_names].copy()

        # Drop initial warm-up rows containing NaNs from indicator initialization
        valid_mask = ~X_raw.isna().any(axis=1)
        valid_count = int(valid_mask.sum())

        if valid_count < self.config.min_train_samples:
            raise InsufficientDataError(
                f"Training data has only {valid_count} complete observations after warmup, "
                f"which is less than the required minimum {self.config.min_train_samples}."
            )

        X_valid = X_raw.loc[valid_mask].values

        # 3. Fit Imputer strictly on training data
        if self.config.impute_missing:
            self.imputer = SimpleImputer(strategy="median")
            X_imputed = self.imputer.fit_transform(X_valid)
        else:
            self.imputer = None
            X_imputed = X_valid

        # 4. Fit StandardScaler strictly on training data
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            self.scaler = None
            X_scaled = X_imputed

        # 5. Fit K-Means strictly on training data
        self.model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
        )
        train_clusters = self.model.fit_predict(X_scaled)

        # 6. Analyze and Interpret Clusters from Training Period
        valid_df = features_df.loc[valid_mask].copy()
        valid_df["regime_cluster"] = train_clusters

        self.cluster_summaries = self._generate_cluster_summaries(valid_df)

        # 7. Record Model Provenance Metadata
        train_start = str(features_df[timestamp_col].iloc[0]) if timestamp_col in features_df.columns else None
        train_end = str(features_df[timestamp_col].iloc[-1]) if timestamp_col in features_df.columns else None

        self.metadata = RegimeModelMetadata(
            model_version=self.config.model_version,
            n_clusters=self.config.n_clusters,
            feature_names=list(self.config.feature_names),
            random_state=self.config.random_state,
            training_sample_count=valid_count,
            train_start=train_start,
            train_end=train_end,
            cluster_summaries=[s.to_dict() for s in self.cluster_summaries],
        )

        self.is_fitted = True
        logger.info(
            "MarketRegimeDetector fitted successfully on %d observations (%d clusters).",
            valid_count,
            self.config.n_clusters,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for observations using strictly learned training parameters.
        
        Warm-up rows containing uncomputable NaNs are assigned cluster -1 (WARMUP).
        
        Args:
            df: DataFrame containing price series or prepared features.
            
        Returns:
            1D numpy array of integer cluster assignments.
            
        Raises:
            ModelNotFittedError: If detector has not been fitted.
        """
        if not self.is_fitted or self.model is None:
            raise ModelNotFittedError("MarketRegimeDetector must be fitted before calling predict().")

        features_df = compute_scale_free_regime_features(df, self.config)
        X_raw = features_df[self.config.feature_names].values

        n_samples = len(X_raw)
        labels = np.full(n_samples, -1, dtype=np.int32)

        # Identify rows with valid non-NaN features
        valid_mask = ~np.isnan(X_raw).any(axis=1)
        if not np.any(valid_mask):
            return labels

        X_valid = X_raw[valid_mask]

        # Apply strictly training-fitted transformations
        if self.imputer is not None:
            X_valid = self.imputer.transform(X_valid)
        if self.scaler is not None:
            X_valid = self.scaler.transform(X_valid)

        predicted_clusters = self.model.predict(X_valid)
        labels[valid_mask] = predicted_clusters
        return labels

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign regime clusters and descriptive labels to a new DataFrame copy.
        
        Preserves original input DataFrame immutably.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Independent copy with 'regime_cluster' and 'regime_label' columns.
        """
        labels = self.predict(df)
        out = df.copy()
        out["regime_cluster"] = labels

        # Map cluster ID to descriptive interpreted label
        label_map: Dict[int, str] = {s.cluster_id: s.interpreted_label for s in self.cluster_summaries}
        label_map[-1] = "WARMUP_UNASSIGNED"

        out["regime_label"] = out["regime_cluster"].map(label_map).fillna("UNKNOWN")
        return out

    def predict_single_bar(self, features: Dict[str, float]) -> int:
        """Predict market regime for a single bar feature vector (zero-lookahead inference).
        
        Args:
            features: Dictionary containing required feature name -> float values.
            
        Returns:
            Integer cluster ID, or -1 if features are invalid.
        """
        if not self.is_fitted or self.model is None:
            raise ModelNotFittedError("MarketRegimeDetector must be fitted before predict_single_bar().")

        vec: List[float] = []
        for name in self.config.feature_names:
            if name not in features or features[name] is None or np.isnan(features[name]):
                return -1
            vec.append(float(features[name]))

        X = np.array([vec], dtype=np.float64)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return int(self.model.predict(X)[0])

    def fit_predict_splits(
        self,
        df: pd.DataFrame,
        splits: Optional[ChronologicalSplits] = None,
        timestamp_col: str = "timestamp",
    ) -> RegimeOutput:
        """Execute complete, leakage-safe chronological partitioning and regime labeling.
        
        Flow:
            1. Partition dataset chronologically: Train (70%), Val (20%), Test (10%).
            2. Fit imputer, scaler, and K-Means STRICTLY on Train split.
            3. Transform Validation and Test splits using Train parameters.
            4. Generate labeled datasets and comprehensive audit metadata.
            
        Args:
            df: Complete historical dataset.
            splits: Optional pre-existing ChronologicalSplits.
            timestamp_col: Name of timestamp column.
            
        Returns:
            RegimeOutput dataclass.
        """
        # 1. Chronological Splitting
        if splits is None:
            splits = chronological_split(
                df=df,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                timestamp_col=timestamp_col,
                embargo_bars=self.config.embargo_bars,
            )

        # 2. Strict Training-Only Fit
        self.fit(splits.train, timestamp_col=timestamp_col)

        # 3. Predict Partitions Without Refitting
        train_labeled = self.predict_df(splits.train)
        val_labeled = self.predict_df(splits.validation)
        test_labeled = self.predict_df(splits.test)

        # Full combined output with labels
        full_labeled = self.predict_df(df)

        # 4. Enrich Metadata with Split Temporal Intervals
        if self.metadata is not None:
            self.metadata.val_start = str(splits.validation[timestamp_col].iloc[0]) if len(splits.validation) > 0 else None
            self.metadata.val_end = str(splits.validation[timestamp_col].iloc[-1]) if len(splits.validation) > 0 else None
            self.metadata.test_start = str(splits.test[timestamp_col].iloc[0]) if len(splits.test) > 0 else None
            self.metadata.test_end = str(splits.test[timestamp_col].iloc[-1]) if len(splits.test) > 0 else None

        return RegimeOutput(
            labeled_df=full_labeled,
            train_df=train_labeled,
            val_df=val_labeled,
            test_df=test_labeled,
            metadata=self.metadata,
            cluster_summaries=list(self.cluster_summaries),
        )

    def _generate_cluster_summaries(self, df_with_clusters: pd.DataFrame) -> List[ClusterSummary]:
        """Empirically characterize clusters from training observations.
        
        Labels are systematically derived based on cluster mean returns and volatility
        relative to the overall training dataset baseline.
        """
        summaries: List[ClusterSummary] = []
        total_obs = len(df_with_clusters)
        if total_obs == 0:
            return summaries

        # Determine baseline characteristics across training set
        ret_col = "simple_return" if "simple_return" in df_with_clusters.columns else df_with_clusters.columns[0]
        vol_col = f"volatility_{self.config.volatility_window}"
        if vol_col not in df_with_clusters.columns:
            # Fallback to any column matching volatility or simple_return
            vol_candidates = [c for c in df_with_clusters.columns if "volatility" in c]
            vol_col = vol_candidates[0] if vol_candidates else ret_col

        mom_col = "momentum_rsi"
        if mom_col not in df_with_clusters.columns:
            mom_candidates = [c for c in df_with_clusters.columns if "momentum" in c or "rsi" in c]
            mom_col = mom_candidates[0] if mom_candidates else ret_col

        base_mean_ret = float(df_with_clusters[ret_col].mean())
        base_mean_vol = float(df_with_clusters[vol_col].mean()) if vol_col in df_with_clusters.columns else 0.0

        for cid in sorted(df_with_clusters["regime_cluster"].unique()):
            c_data = df_with_clusters[df_with_clusters["regime_cluster"] == cid]
            count = len(c_data)
            pct = count / total_obs

            mean_ret = float(c_data[ret_col].mean()) if ret_col in c_data.columns else 0.0
            median_ret = float(c_data[ret_col].median()) if ret_col in c_data.columns else 0.0
            mean_vol = float(c_data[vol_col].mean()) if vol_col in c_data.columns else 0.0
            mean_mom = float(c_data[mom_col].mean()) if mom_col in c_data.columns else 0.0

            # Characterize return tone
            if mean_ret > base_mean_ret + 0.0005:
                ret_desc = "Bullish/Positive"
            elif mean_ret < base_mean_ret - 0.0005:
                ret_desc = "Bearish/Negative"
            else:
                ret_desc = "Neutral/Sideways"

            # Characterize volatility tone
            if mean_vol > base_mean_vol * 1.15:
                vol_desc = "High Volatility"
            elif mean_vol < base_mean_vol * 0.85:
                vol_desc = "Low Volatility"
            else:
                vol_desc = "Moderate Volatility"

            interpreted_label = f"Regime {cid} ({vol_desc}, {ret_desc})"

            # Symbol distribution breakdown
            sym_dist: Dict[str, int] = {}
            if "symbol" in c_data.columns:
                sym_dist = c_data["symbol"].value_counts().to_dict()

            summary = ClusterSummary(
                cluster_id=int(cid),
                count=count,
                pct_observations=pct,
                mean_return=mean_ret,
                median_return=median_ret,
                mean_volatility=mean_vol,
                mean_momentum=mean_mom,
                interpreted_label=interpreted_label,
                symbol_distribution=sym_dist,
            )
            summaries.append(summary)

        return summaries

    def save(self, filepath: str) -> None:
        """Serialize fitted model pipeline and metadata to disk via joblib."""
        if not self.is_fitted:
            raise ModelNotFittedError("Cannot save unfitted MarketRegimeDetector.")

        state = {
            "config": self.config,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "cluster_summaries": self.cluster_summaries,
            "metadata": self.metadata,
        }
        joblib.dump(state, filepath)
        logger.info("Saved MarketRegimeDetector to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> MarketRegimeDetector:
        """Load serialized MarketRegimeDetector from disk."""
        state = joblib.load(filepath)
        detector = cls(config=state["config"])
        detector.imputer = state["imputer"]
        detector.scaler = state["scaler"]
        detector.model = state["model"]
        detector.is_fitted = state["is_fitted"]
        detector.cluster_summaries = state["cluster_summaries"]
        detector.metadata = state["metadata"]
        logger.info("Loaded MarketRegimeDetector from %s", filepath)
        return detector
