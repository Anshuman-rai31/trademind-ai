"""Real-Time RL Trading Signal Generation Service (Phase 12).

Orchestrates live PPO policy inference, exact 12-d observation vector construction,
timeframe compatibility validation, warmup sufficiency checking, pre-trade risk evaluation,
duplicate candle prevention, audit persistence, and multi-symbol state isolation.
"""
from __future__ import annotations

import collections
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..data.validator import validate_symbol
from ..db import db, safe_json_dumps, safe_json_loads, safe_serialize
from ..models.market_candle import MarketCandle
from ..models.model_registry import RegisteredModel
from ..models.paper_trading import PaperPortfolio, PaperPosition
from ..models.trading_signal import RLTradingSignal
from ..realtime.candle_builder import Candle
from ..realtime.feature_pipeline import EXPECTED_OBS_DIM, MIN_WARMUP_BARS, LiveFeaturePipeline
from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..risk.models import InterventionStatus
from ..services.market_data_service import MarketDataService
from ..services.model_registry_service import ModelRegistryService
from ..simulation.models import Order, OrderSide
from ..simulation.portfolio import Portfolio

logger = logging.getLogger(__name__)

SUPPORTED_NSE_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
]

ACTION_MAP = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
}


class RealtimeSignalService:
    """Institutional real-time RL trading signal generation engine."""

    _instance: Optional[RealtimeSignalService] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        registry_service: Optional[ModelRegistryService] = None,
        data_service: Optional[MarketDataService] = None,
        risk_manager: Optional[RiskManager] = None,
        feature_pipeline: Optional[LiveFeaturePipeline] = None,
        trained_model_timeframe: str = "1d",
        min_warmup_bars: int = MIN_WARMUP_BARS,
    ):
        self.registry_service = registry_service or ModelRegistryService()
        self.data_service = data_service or MarketDataService()
        self.risk_manager = risk_manager or RiskManager(config=RiskConfig(enabled=True))
        self.trained_model_timeframe = trained_model_timeframe.lower()
        self.min_warmup_bars = min_warmup_bars

        self.feature_pipeline = feature_pipeline or LiveFeaturePipeline(
            trained_model_timeframe=self.trained_model_timeframe,
            min_warmup_bars=self.min_warmup_bars,
        )

        self._lock = threading.Lock()
        # In-memory deduplication cache: (symbol, interval, source_candle_timestamp)
        self._processed_keys_queue: Deque[Tuple[str, str, str]] = collections.deque(maxlen=2000)
        self._processed_keys_set: Set[Tuple[str, str, str]] = set()

        # In-memory history buffer per (symbol, interval) to avoid continuous DB round-trips
        self._candle_buffers: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=300)
        )

        # Downstream signal listeners
        self._signal_listeners: List[Callable[[Dict[str, Any]], None]] = []

        # Service metrics / telemetry
        self._telemetry = {
            "total_signals_generated": 0,
            "total_valid_signals": 0,
            "total_risk_blocked": 0,
            "total_timeframe_mismatches": 0,
            "total_rejected": 0,
            "total_duplicates_prevented": 0,
            "last_signal_time": None,
            "last_error": None,
        }

    @classmethod
    def get_instance(
        cls,
        registry_service: Optional[ModelRegistryService] = None,
        data_service: Optional[MarketDataService] = None,
        risk_manager: Optional[RiskManager] = None,
        feature_pipeline: Optional[LiveFeaturePipeline] = None,
    ) -> RealtimeSignalService:
        """Thread-safe singleton accessor."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(
                    registry_service=registry_service,
                    data_service=data_service,
                    risk_manager=risk_manager,
                    feature_pipeline=feature_pipeline,
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (primarily for isolated test fixtures)."""
        with cls._singleton_lock:
            cls._instance = None

    def register_signal_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback invoked whenever a signal is generated."""
        with self._lock:
            if listener not in self._signal_listeners:
                self._signal_listeners.append(listener)

    def on_completed_candle(self, candle: Candle) -> Optional[Dict[str, Any]]:
        """Consumer callback suitable for registering directly with CandleBuilderService."""
        candle_dict = candle.to_dict()
        return self.process_completed_candle(
            symbol=candle.symbol,
            candle=candle_dict,
            interval=candle.interval,
        )

    def process_completed_candle(
        self,
        symbol: str,
        candle: Dict[str, Any],
        interval: str = "5m",
        portfolio_id: Optional[int] = None,
        data_freshness: str = "FRESH",
    ) -> Dict[str, Any]:
        """Process a completed candle to evaluate PPO model policy and generate signal."""
        from flask import has_app_context
        from ..db import get_flask_app

        if not has_app_context():
            app = get_flask_app()
            if app:
                with app.app_context():
                    return self._process_completed_candle_impl(
                        symbol=symbol,
                        candle=candle,
                        interval=interval,
                        portfolio_id=portfolio_id,
                        data_freshness=data_freshness,
                    )
        return self._process_completed_candle_impl(
            symbol=symbol,
            candle=candle,
            interval=interval,
            portfolio_id=portfolio_id,
            data_freshness=data_freshness,
        )

    def _process_completed_candle_impl(
        self,
        symbol: str,
        candle: Dict[str, Any],
        interval: str = "5m",
        portfolio_id: Optional[int] = None,
        data_freshness: str = "FRESH",
    ) -> Dict[str, Any]:
        """Internal implementation of candle processing and signal evaluation."""
        try:
            clean_sym = validate_symbol(symbol)
        except Exception as exc:
            logger.error("Invalid symbol format for signal generation: %s", exc)
            return self._build_error_response(symbol, "INVALID_SYMBOL", str(exc))

        clean_interval = str(interval).strip().lower()

        # 1. Validate Candle Completeness and Fields
        if not candle or not isinstance(candle, dict):
            return self._build_error_response(clean_sym, "MALFORMED_CANDLE", "Candle must be a valid dictionary.")

        is_complete = candle.get("is_complete", True)
        if not is_complete:
            logger.warning("[%s] Incomplete candle received; signal generation requires closed bar.", clean_sym)
            return self._build_error_response(
                clean_sym,
                "INCOMPLETE_CANDLE",
                "Signal generation requires closed, completed candle bars.",
            )

        candle_timestamp = str(candle.get("start_time") or candle.get("timestamp") or "")
        if not candle_timestamp:
            return self._build_error_response(clean_sym, "MISSING_TIMESTAMP", "Candle missing start_time/timestamp.")

        close_price = candle.get("close")
        if close_price is None or not isinstance(close_price, (int, float)) or close_price <= 0:
            return self._build_error_response(clean_sym, "INVALID_PRICE", f"Invalid close price: {close_price}")

        # 2. Duplicate Prevention
        key = (clean_sym, clean_interval, candle_timestamp)
        with self._lock:
            if key in self._processed_keys_set:
                self._telemetry["total_duplicates_prevented"] += 1
                logger.info("[%s] Duplicate completed candle %s already processed; skipping signal generation.", clean_sym, key)
                existing_sig = RLTradingSignal.query.filter_by(
                    symbol=clean_sym,
                    interval=clean_interval,
                    source_candle_timestamp=candle_timestamp,
                ).first()
                if existing_sig:
                    return existing_sig.to_dict()
                return self._build_error_response(clean_sym, "DUPLICATE_CANDLE", f"Candle {key} already processed.")

            # Check database for persistent deduplication
            existing_sig = RLTradingSignal.query.filter_by(
                symbol=clean_sym,
                interval=clean_interval,
                source_candle_timestamp=candle_timestamp,
            ).first()
            if existing_sig:
                self._mark_key_processed(key)
                self._telemetry["total_duplicates_prevented"] += 1
                return existing_sig.to_dict()

        # 3. Buffer and Warmup History Assembly
        candles_history = self._assemble_warmup_history(clean_sym, clean_interval, candle)

        # 4. Fetch Portfolio State for the 5 portfolio features
        portfolio_state, port_obj = self._get_portfolio_state(portfolio_id, clean_sym, float(close_price))

        # 5. Feature Engineering and Observation Generation
        obs_result = self.feature_pipeline.generate_live_observation(
            candles_history=candles_history,
            portfolio_state=portfolio_state,
            candle_interval=clean_interval,
            symbol=clean_sym,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        candle_start = str(candle.get("start_time") or candle_timestamp)
        candle_end = str(candle.get("end_time") or candle_timestamp)

        # 6. Check Timeframe Mismatch
        is_mismatch = (clean_interval != self.trained_model_timeframe) or obs_result.get("timeframe_mismatch", False)
        if is_mismatch:
            # Safe handling: do not pretend model can run on intraday data
            with self._lock:
                self._telemetry["total_timeframe_mismatches"] += 1
                self._telemetry["total_signals_generated"] += 1
                self._telemetry["last_signal_time"] = now_iso

            mismatch_reason = obs_result.get("reason") or (
                f"Timeframe mismatch: Active PPO model was trained on daily ({self.trained_model_timeframe}) candles, "
                f"but received intraday ({clean_interval}) candle. Direct model inference on intraday bars is disabled "
                f"to prevent severe policy distribution shift."
            )

            signal_record = RLTradingSignal(
                symbol=clean_sym,
                portfolio_id=portfolio_id or (port_obj.id if port_obj else None),
                interval=clean_interval,
                action="HOLD",
                proposed_action=None,
                effective_action="HOLD",
                confidence=None,
                close_price=float(close_price),
                source_candle_timestamp=candle_timestamp,
                source_candle_start=candle_start,
                source_candle_end=candle_end,
                timestamp=now_iso,
                model_id="ppo_daily_policy",
                model_version="1.0.0",
                data_freshness=data_freshness,
                status="TIMEFRAME_MISMATCH",
                rejection_reason=mismatch_reason,
                observation_json=safe_json_dumps(obs_result.get("observation")),
                features_json=safe_json_dumps(obs_result.get("features")),
                risk_evaluation_json=safe_json_dumps({"status": "WITHHELD", "reason": "Timeframe mismatch"}),
            )
            return self._persist_and_dispatch(signal_record, key)

        # 7. Check Feature Calculation Readiness (Warmup or NaNs)
        if not obs_result.get("ready", False):
            with self._lock:
                self._telemetry["total_rejected"] += 1
                self._telemetry["total_signals_generated"] += 1
                self._telemetry["last_signal_time"] = now_iso

            signal_record = RLTradingSignal(
                symbol=clean_sym,
                portfolio_id=portfolio_id or (port_obj.id if port_obj else None),
                interval=clean_interval,
                action="HOLD",
                proposed_action=None,
                effective_action="HOLD",
                confidence=None,
                close_price=float(close_price),
                source_candle_timestamp=candle_timestamp,
                source_candle_start=candle_start,
                source_candle_end=candle_end,
                timestamp=now_iso,
                model_id="ppo_daily_policy",
                model_version="1.0.0",
                data_freshness=data_freshness,
                status="REJECTED",
                rejection_reason=obs_result.get("reason", "Feature generation failed."),
                observation_json=None,
                features_json=safe_json_dumps(obs_result.get("features")),
                risk_evaluation_json=None,
            )
            return self._persist_and_dispatch(signal_record, key)

        # 8. Load Active Approved PPO Model from Model Registry
        self.registry_service.sync_local_artifacts()
        try:
            model, model_rec = self.registry_service.load_model_for_inference(clean_sym)
        except Exception as exc:
            logger.error("[%s] Failed to load model for inference: %s", clean_sym, exc)
            with self._lock:
                self._telemetry["total_rejected"] += 1
                self._telemetry["total_signals_generated"] += 1
                self._telemetry["last_signal_time"] = now_iso

            signal_record = RLTradingSignal(
                symbol=clean_sym,
                portfolio_id=portfolio_id or (port_obj.id if port_obj else None),
                interval=clean_interval,
                action="HOLD",
                proposed_action=None,
                effective_action="HOLD",
                confidence=None,
                close_price=float(close_price),
                source_candle_timestamp=candle_timestamp,
                source_candle_start=candle_start,
                source_candle_end=candle_end,
                timestamp=now_iso,
                status="REJECTED",
                rejection_reason=f"Model loading error: {exc}",
                observation_json=safe_json_dumps(obs_result.get("observation")),
                features_json=safe_json_dumps(obs_result.get("features")),
            )
            return self._persist_and_dispatch(signal_record, key)

        # 9. Deterministic PPO Policy Prediction
        raw_obs = obs_result.get("observation")
        obs_array = np.array(raw_obs, dtype=np.float32)

        try:
            raw_action, _ = model.predict(obs_array, deterministic=True)
            action_int = int(raw_action)
            proposed_action_name = ACTION_MAP.get(action_int, "HOLD")
        except Exception as exc:
            logger.error("[%s] Inference execution error: %s", clean_sym, exc)
            with self._lock:
                self._telemetry["total_rejected"] += 1
                self._telemetry["total_signals_generated"] += 1
                self._telemetry["last_signal_time"] = now_iso

            signal_record = RLTradingSignal(
                symbol=clean_sym,
                portfolio_id=portfolio_id or (port_obj.id if port_obj else None),
                interval=clean_interval,
                action="HOLD",
                proposed_action=None,
                effective_action="HOLD",
                confidence=None,
                close_price=float(close_price),
                source_candle_timestamp=candle_timestamp,
                source_candle_start=candle_start,
                source_candle_end=candle_end,
                timestamp=now_iso,
                model_id=model_rec.model_id,
                model_version=model_rec.version,
                model_checksum=model_rec.artifact_checksum,
                status="REJECTED",
                rejection_reason=f"Inference execution failed: {exc}",
                observation_json=safe_json_dumps(raw_obs),
                features_json=safe_json_dumps(obs_result.get("features")),
            )
            return self._persist_and_dispatch(signal_record, key)

        # 10. Pre-Trade Risk Management Gate
        risk_status = "VALID"
        effective_action_name = proposed_action_name
        rejection_reason = None
        risk_eval_details = {
            "status": "APPROVED",
            "proposed_action": proposed_action_name,
            "details": "Model action approved by risk policy",
        }

        if proposed_action_name in ("BUY", "SELL"):
            risk_eval = self._evaluate_risk(
                symbol=clean_sym,
                proposed_action=proposed_action_name,
                close_price=float(close_price),
                timestamp=now_iso,
                port_obj=port_obj,
            )
            risk_status = risk_eval["status"]
            effective_action_name = risk_eval["effective_action"]
            rejection_reason = risk_eval.get("reason")
            risk_eval_details = risk_eval

        # Update telemetry
        with self._lock:
            self._telemetry["total_signals_generated"] += 1
            if risk_status == "RISK_BLOCKED":
                self._telemetry["total_risk_blocked"] += 1
            else:
                self._telemetry["total_valid_signals"] += 1
            self._telemetry["last_signal_time"] = now_iso

        signal_record = RLTradingSignal(
            symbol=clean_sym,
            portfolio_id=portfolio_id or (port_obj.id if port_obj else None),
            interval=clean_interval,
            action=effective_action_name,
            proposed_action=proposed_action_name,
            effective_action=effective_action_name,
            confidence=None,  # No fabricated confidence
            close_price=float(close_price),
            source_candle_timestamp=candle_timestamp,
            source_candle_start=candle_start,
            source_candle_end=candle_end,
            timestamp=now_iso,
            model_id=model_rec.model_id,
            model_version=model_rec.version,
            model_checksum=model_rec.artifact_checksum,
            data_freshness=data_freshness,
            status=risk_status,
            rejection_reason=rejection_reason,
            observation_json=safe_json_dumps(raw_obs),
            features_json=safe_json_dumps(obs_result.get("features")),
            risk_evaluation_json=safe_json_dumps(risk_eval_details),
        )

        return self._persist_and_dispatch(signal_record, key)

    def _assemble_warmup_history(
        self,
        symbol: str,
        interval: str,
        current_candle: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Assemble chronological candle sequence ensuring >= 50 completed bars without lookahead."""
        key = (symbol, interval)
        with self._lock:
            buf = self._candle_buffers[key]

        # If buffer is insufficient, fetch historical warmup from storage
        if len(buf) < self.min_warmup_bars:
            self._warmup_buffer_from_storage(symbol, interval)

        # Append current candle if not already the latest
        with self._lock:
            if not buf or buf[-1].get("start_time") != current_candle.get("start_time"):
                buf.append(current_candle)
            return list(buf)

    def _warmup_buffer_from_storage(self, symbol: str, interval: str) -> None:
        """Prefill candle buffer from persistent database / CSV datasets."""
        key = (symbol, interval)
        loaded_candles: List[Dict[str, Any]] = []

        if interval == "1d":
            # Load from historical data service (daily bars)
            try:
                hist_res = self.data_service.get_historical_bars(symbol, limit=200)
                if hist_res and hist_res[0]:
                    bars, _ = hist_res
                    for b in bars:
                        ts = str(b.get("timestamp"))
                        loaded_candles.append({
                            "symbol": symbol,
                            "interval": "1d",
                            "start_time": ts,
                            "end_time": ts,
                            "open": float(b.get("open", 0.0)),
                            "high": float(b.get("high", 0.0)),
                            "low": float(b.get("low", 0.0)),
                            "close": float(b.get("close", 0.0)),
                            "volume": int(b.get("volume", 0)),
                            "is_complete": True,
                        })
            except Exception as exc:
                logger.debug("[%s] Warmup from historical CSV failed: %s", symbol, exc)

        # Fallback / augment with MarketCandle SQLite table
        if len(loaded_candles) < self.min_warmup_bars:
            try:
                records = (
                    MarketCandle.query.filter_by(symbol=symbol, interval=interval, is_complete=True)
                    .order_by(MarketCandle.start_time.desc())
                    .limit(self.min_warmup_bars + 20)
                    .all()
                )
                if records:
                    db_candles = [r.to_dict() for r in reversed(records)]
                    # Merge uniquely by start_time
                    existing_starts = {c["start_time"] for c in loaded_candles}
                    for dbc in db_candles:
                        if dbc["start_time"] not in existing_starts:
                            loaded_candles.append(dbc)
            except Exception as exc:
                logger.debug("[%s] Warmup from MarketCandle table failed: %s", symbol, exc)

        if loaded_candles:
            # Sort chronologically
            loaded_candles.sort(key=lambda x: str(x.get("start_time")))
            with self._lock:
                for c in loaded_candles:
                    if not self._candle_buffers[key] or self._candle_buffers[key][-1].get("start_time") != c.get("start_time"):
                        self._candle_buffers[key].append(c)

    def _get_portfolio_state(
        self,
        portfolio_id: Optional[int],
        symbol: str,
        close_price: float,
    ) -> Tuple[Dict[str, Any], Optional[PaperPortfolio]]:
        """Retrieve portfolio state for the 5 observation features."""
        try:
            port = None
            if portfolio_id:
                port = PaperPortfolio.query.get(portfolio_id)
            if not port:
                port = PaperPortfolio.query.filter_by(name="default").first()

            if port:
                pos = PaperPosition.query.filter_by(portfolio_id=port.id, symbol=symbol).first()
                holdings = float(pos.quantity) if pos else 0.0
                cost_basis = float(pos.average_entry_price) if pos else 0.0
                return {
                    "cash": float(port.cash),
                    "holdings": holdings,
                    "cost_basis": cost_basis,
                    "peak_value": float(port.peak_value or port.initial_cash),
                }, port
        except Exception as exc:
            logger.debug("[%s] Could not read portfolio state from DB: %s", symbol, exc)

        # Default simulated state
        return {
            "cash": 100000.0,
            "holdings": 0.0,
            "cost_basis": 0.0,
            "peak_value": 100000.0,
        }, None

    def _evaluate_risk(
        self,
        symbol: str,
        proposed_action: str,
        close_price: float,
        timestamp: str,
        port_obj: Optional[PaperPortfolio],
    ) -> Dict[str, Any]:
        """Perform pre-trade risk evaluation against RiskManager."""
        try:
            port_cash = float(port_obj.cash) if port_obj else 100000.0
            port_peak = float(port_obj.peak_value) if port_obj else 100000.0
            pos = PaperPosition.query.filter_by(portfolio_id=port_obj.id, symbol=symbol).first() if port_obj else None
            current_holding = float(pos.quantity) if pos else 0.0
            avg_entry = float(pos.average_entry_price) if pos else 0.0

            allocation_pct = 0.95
            target_capital = port_cash * allocation_pct
            proposed_qty = np.floor(target_capital / close_price) if proposed_action == "BUY" else current_holding

            if proposed_qty <= 0:
                return {
                    "status": "VALID",
                    "effective_action": "HOLD",
                    "reason": "Insufficient capital or zero holdings to execute action.",
                    "details": "Quantity evaluated to 0",
                }

            order_side = OrderSide.BUY if proposed_action == "BUY" else OrderSide.SELL
            order = Order(
                symbol=symbol,
                side=order_side,
                quantity=float(proposed_qty),
                created_at=pd.Timestamp(timestamp),
            )

            sim_portfolio = Portfolio(initial_cash=port_peak)
            sim_portfolio.cash = port_cash
            sim_portfolio.peak_value = port_peak
            if current_holding > 0:
                sim_portfolio.holdings[symbol] = current_holding
                sim_portfolio.cost_basis[symbol] = avg_entry

            approved_order, intervention = self.risk_manager.evaluate_proposed_order(
                order=order,
                portfolio=sim_portfolio,
                execution_price=close_price,
                timestamp=timestamp,
            )

            if intervention is not None:
                if approved_order is None or approved_order.quantity <= 0:
                    return {
                        "status": "RISK_BLOCKED",
                        "effective_action": "HOLD",
                        "proposed_action": proposed_action,
                        "rule_breached": str(intervention.reason.value if hasattr(intervention.reason, "value") else intervention.reason),
                        "reason": f"Risk blocked: {intervention.details}",
                        "details": intervention.details,
                    }
                else:
                    return {
                        "status": "RISK_MODIFIED",
                        "effective_action": proposed_action,
                        "proposed_action": proposed_action,
                        "modified_quantity": approved_order.quantity,
                        "rule_breached": str(intervention.reason.value if hasattr(intervention.reason, "value") else intervention.reason),
                        "reason": f"Risk modified: {intervention.details}",
                        "details": intervention.details,
                    }

            return {
                "status": "VALID",
                "effective_action": proposed_action,
                "proposed_action": proposed_action,
                "details": "Order approved by pre-trade risk evaluation",
            }
        except Exception as exc:
            logger.warning("[%s] Risk evaluation exception: %s", symbol, exc)
            return {
                "status": "VALID",
                "effective_action": proposed_action,
                "proposed_action": proposed_action,
                "details": f"Risk evaluation fallback: {exc}",
            }

    def _persist_and_dispatch(
        self,
        signal_record: RLTradingSignal,
        key: Tuple[str, str, str],
    ) -> Dict[str, Any]:
        """Save signal record to SQLite and notify registered listeners."""
        try:
            db.session.add(signal_record)
            db.session.commit()
            self._mark_key_processed(key)
        except Exception as exc:
            db.session.rollback()
            logger.error("Failed to persist RLTradingSignal: %s", exc)

        res_dict = signal_record.to_dict()

        # Notify downstream listeners outside database transaction
        with self._lock:
            listeners = list(self._signal_listeners)

        for cb in listeners:
            try:
                cb(res_dict)
            except Exception as exc:
                logger.error("Error in signal listener callback: %s", exc)

        return res_dict

    def _mark_key_processed(self, key: Tuple[str, str, str]) -> None:
        """Mark candle key as processed in bounded memory set."""
        with self._lock:
            if key not in self._processed_keys_set:
                self._processed_keys_queue.append(key)
                self._processed_keys_set.add(key)
                # Keep set bounded to maxlen
                if len(self._processed_keys_set) > len(self._processed_keys_queue) + 200:
                    self._processed_keys_set = set(self._processed_keys_queue)

    def _build_error_response(self, symbol: str, code: str, message: str) -> Dict[str, Any]:
        """Generate standardized rejected signal dictionary for client errors."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._telemetry["total_rejected"] += 1
            self._telemetry["last_error"] = f"{code}: {message}"

        return {
            "symbol": symbol,
            "action": "HOLD",
            "proposed_action": None,
            "effective_action": "HOLD",
            "confidence": None,
            "status": "REJECTED",
            "rejection_reason": f"{code}: {message}",
            "timestamp": now_iso,
            "observation": None,
            "features": None,
            "risk_evaluation": None,
        }

    # =========================================================================
    # Query API Methods
    # =========================================================================

    def get_latest_signals(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve the latest signal per symbol, or for a single symbol."""
        if symbol:
            try:
                clean_sym = validate_symbol(symbol)
                latest = (
                    RLTradingSignal.query.filter_by(symbol=clean_sym)
                    .order_by(RLTradingSignal.id.desc())
                    .first()
                )
                return {clean_sym: latest.to_dict() if latest else None}
            except Exception:
                return {}

        results: Dict[str, Any] = {}
        for sym in SUPPORTED_NSE_SYMBOLS:
            latest = (
                RLTradingSignal.query.filter_by(symbol=sym)
                .order_by(RLTradingSignal.id.desc())
                .first()
            )
            if not latest:
                # Automatically evaluate latest historical daily bar if no signal exists yet
                try:
                    hist = self.data_service.get_historical_bars(sym, limit=1)
                    if hist and hist[0]:
                        bar = hist[0][0]
                        candle_input = {
                            "symbol": sym,
                            "interval": "1d",
                            "start_time": str(bar.get("timestamp")),
                            "end_time": str(bar.get("timestamp")),
                            "open": float(bar.get("open", 0.0)),
                            "high": float(bar.get("high", 0.0)),
                            "low": float(bar.get("low", 0.0)),
                            "close": float(bar.get("close", 0.0)),
                            "volume": int(bar.get("volume", 0)),
                            "is_complete": True,
                        }
                        self.process_completed_candle(sym, candle_input, interval="1d")
                        latest = (
                            RLTradingSignal.query.filter_by(symbol=sym)
                            .order_by(RLTradingSignal.id.desc())
                            .first()
                        )
                except Exception as exc:
                    logger.debug("[%s] Auto-evaluation on first observation: %s", sym, exc)

            results[sym] = latest.to_dict() if latest else None
        return results

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        interval: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Retrieve paginated signals with optional filtering."""
        query = RLTradingSignal.query

        if symbol:
            try:
                clean_sym = validate_symbol(symbol)
                query = query.filter_by(symbol=clean_sym)
            except Exception:
                pass

        if action:
            query = query.filter_by(action=str(action).strip().upper())

        if status:
            query = query.filter_by(status=str(status).strip().upper())

        if interval:
            query = query.filter_by(interval=str(interval).strip().lower())

        total = query.count()
        safe_page = max(1, page)
        safe_size = min(max(1, page_size), 100)
        offset = (safe_page - 1) * safe_size

        records = query.order_by(RLTradingSignal.id.desc()).offset(offset).limit(safe_size).all()

        return {
            "signals": [r.to_dict() for r in records],
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
            "total_pages": int(np.ceil(total / safe_size)) if total > 0 else 1,
        }

    def get_signal_by_id(self, signal_id_or_int: str | int) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific signal record."""
        record = None
        if isinstance(signal_id_or_int, int) or (isinstance(signal_id_or_int, str) and signal_id_or_int.isdigit()):
            record = db.session.get(RLTradingSignal, int(signal_id_or_int))
        else:
            record = RLTradingSignal.query.filter_by(signal_id=str(signal_id_or_int)).first()

        return record.to_dict() if record else None

    def get_service_health(self) -> Dict[str, Any]:
        """Retrieve signal service readiness, telemetry, and active model status."""
        active_models: Dict[str, Any] = {}
        for sym in SUPPORTED_NSE_SYMBOLS:
            act = self.registry_service.get_active_model(sym)
            if not act:
                act = (
                    RegisteredModel.query.filter_by(symbol=sym, status="approved")
                    .order_by(RegisteredModel.id.desc())
                    .first()
                )
            active_models[sym] = {
                "model_id": act.model_id if act else None,
                "version": act.version if act else None,
                "status": act.status if act else "not_configured",
                "checksum": act.artifact_checksum if act else None,
            }

        return {
            "service": "RealtimeSignalService",
            "status": "ONLINE",
            "supported_symbols": list(SUPPORTED_NSE_SYMBOLS),
            "trained_model_timeframe": self.trained_model_timeframe,
            "active_models": active_models,
            "telemetry": dict(self._telemetry),
            "memory_cache": {
                "dedup_keys_count": len(self._processed_keys_set),
                "candle_buffers_count": len(self._candle_buffers),
            },
        }
