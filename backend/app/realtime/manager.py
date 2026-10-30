"""Real-Time Market Data Manager & Orchestrator (Phase 11A - 11G).

Coordinates provider lifecycle, live tick ingestion & validation (11C),
1m/5m OHLCV candle construction (11D), PPO feature engineering (11E),
and live dashboard data aggregation (11F) with zero lookahead bias.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from .base_provider import RealtimeDataProvider
from .candle_builder import Candle, CandleBuilderService
from .feature_pipeline import LiveFeaturePipeline
from .ingestion import LiveMarketDataIngestionService
from .kite_provider import KiteRealtimeProvider
from .mock_provider import MockRealtimeProvider
from .models import ConnectionStatus, ConnectionTelemetry, MarketQuote, ProviderConfig
from .subscription import InstrumentActivityStatus, SubscriptionState
from .validator import RealtimeDataValidator

logger = logging.getLogger(__name__)


class RealtimeMarketDataManager:
    """Singleton-capable coordinator for market data streaming, candle building, and feature pipelines."""

    _instance: Optional[RealtimeMarketDataManager] = None
    _lock = threading.Lock()

    def __init__(
        self,
        provider: Optional[RealtimeDataProvider] = None,
        config: Optional[ProviderConfig] = None,
        validator: Optional[RealtimeDataValidator] = None,
        ingestion_service: Optional[LiveMarketDataIngestionService] = None,
        candle_builder: Optional[CandleBuilderService] = None,
        feature_pipeline: Optional[LiveFeaturePipeline] = None,
        signal_service: Optional[Any] = None,
    ):
        self.config = config or ProviderConfig()
        self.validator = validator or RealtimeDataValidator()
        self.provider = provider or self._create_default_provider(self.config, self.validator)

        # Ingestion, Candle Builder, and Feature Pipeline services
        self.ingestion_service = ingestion_service or LiveMarketDataIngestionService(
            validator=self.validator,
            freshness_threshold_seconds=self.config.stale_timeout_seconds,
        )
        self.candle_builder = candle_builder or CandleBuilderService(
            default_interval="5m",
            max_completed_candles=500,
        )
        self.feature_pipeline = feature_pipeline or LiveFeaturePipeline(
            trained_model_timeframe="1d",
        )
        self.signal_service = signal_service

        # Wire pipeline: provider -> ingestion -> candle builder -> signals
        self._wire_subsystems()

        self._active_portfolio_name: Optional[str] = "default"
        self._staleness_checker_thread: Optional[threading.Thread] = None
        self._is_monitoring = False

    def _wire_subsystems(self) -> None:
        """Connect provider callbacks to ingestion and candle construction."""
        self.provider.register_callback(self.ingestion_service.ingest_quote)
        self.ingestion_service.register_consumer(self.candle_builder.on_tick)
        if self.signal_service is not None and hasattr(self.signal_service, "on_completed_candle"):
            self.candle_builder.register_candle_consumer(self.signal_service.on_completed_candle)

    def attach_signal_service(self, signal_service: Any) -> None:
        """Dynamically attach signal generation service to completed candle stream."""
        self.signal_service = signal_service
        if hasattr(signal_service, "on_completed_candle"):
            self.candle_builder.register_candle_consumer(signal_service.on_completed_candle)

    @classmethod
    def get_instance(cls, config: Optional[ProviderConfig] = None) -> RealtimeMarketDataManager:
        """Retrieve thread-safe singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config=config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for test isolation)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect()
                cls._instance = None

    def _create_default_provider(
        self,
        config: ProviderConfig,
        validator: RealtimeDataValidator,
    ) -> RealtimeDataProvider:
        """Instantiate selected provider backend based on configuration."""
        if config.provider_name.lower() == "kite":
            if not config.has_credentials():
                logger.warning(
                    "Kite provider requested but KITE_API_KEY / KITE_ACCESS_TOKEN are not configured in environment. "
                    "Auto-fallback to MockRealtimeProvider with live simulated tick generation."
                )
                return MockRealtimeProvider(config=config, validator=validator, auto_tick_generation=True)
            return KiteRealtimeProvider(config=config, validator=validator)
        return MockRealtimeProvider(config=config, validator=validator, auto_tick_generation=True)

    def set_provider(self, provider: RealtimeDataProvider) -> None:
        """Hot-swap provider instance (e.g. for testing with mock provider)."""
        if self.provider:
            self.provider.disconnect()
        self.provider = provider
        self._wire_subsystems()

    def connect(self) -> bool:
        """Initiate provider connection and start staleness watchdog."""
        ok = self.provider.connect()
        if ok and not self._is_monitoring:
            self._start_staleness_watchdog()
        return ok

    def disconnect(self) -> None:
        """Gracefully disconnect provider and stop watchdog."""
        self._is_monitoring = False
        self.provider.disconnect()

    def reconnect(self) -> bool:
        """Cleanly disconnect and reconnect with subscription reconciliation."""
        self.disconnect()
        return self.connect()

    def get_status(self) -> ConnectionStatus:
        """Get current provider connection status (evaluating staleness)."""
        self.check_staleness()
        return self.provider.status

    def get_telemetry(self) -> ConnectionTelemetry:
        """Retrieve diagnostic telemetry without secrets."""
        self.check_staleness()
        return self.provider.telemetry

    def get_latest_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Retrieve latest validated quote for symbol from ingestion cache."""
        return self.ingestion_service.get_latest_quote(symbol) or self.provider.get_latest_quote(symbol)

    def get_all_quotes(self) -> Dict[str, MarketQuote]:
        """Retrieve all currently cached quotes."""
        quotes = self.ingestion_service.get_all_latest_quotes()
        if not quotes:
            quotes = self.provider.get_all_quotes()
        return quotes

    def get_symbol_freshness(self, symbol: str) -> Dict[str, Any]:
        """Retrieve instrument freshness and latency telemetry."""
        clean_sym = str(symbol).strip().upper()
        f = self.ingestion_service.get_symbol_freshness(clean_sym)
        if f.get("status") == "OFFLINE":
            q = self.provider.get_latest_quote(clean_sym)
            if q is not None:
                self.ingestion_service.ingest_quote(q)
                return self.ingestion_service.get_symbol_freshness(clean_sym)
            elif self.provider.status == ConnectionStatus.CONNECTED:
                if hasattr(self.provider, "emit_mock_ticks_for_subscribed"):
                    quotes = self.provider.emit_mock_ticks_for_subscribed()
                    for quote in quotes:
                        self.ingestion_service.ingest_quote(quote)
                    return self.ingestion_service.get_symbol_freshness(clean_sym)
        return f

    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe provider to a list of symbols."""
        return self.provider.subscribe(symbols)

    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe provider from a list of symbols."""
        return self.provider.unsubscribe(symbols)

    def get_subscribed_symbols(self) -> List[str]:
        """List of active subscriptions."""
        return self.provider.subscribed_symbols

    def get_subscription_details(self) -> Dict[str, Any]:
        """Get comprehensive instrument subscription registry and activity statuses."""
        self.check_staleness()
        return self.provider.get_subscription_details()

    def get_health_report(self) -> Dict[str, Any]:
        """Generate high-level health report distinguishing connection failure from inactive markets."""
        self.check_staleness()
        telemetry = self.provider.telemetry
        sub_details = self.get_subscription_details()

        is_connected = telemetry.status == ConnectionStatus.CONNECTED and not telemetry.is_stale

        return {
            "status": telemetry.status.value,
            "provider": telemetry.provider_name,
            "connected": is_connected,
            "is_stale": telemetry.is_stale,
            "last_heartbeat_at": telemetry.last_heartbeat_at,
            "last_message_at": telemetry.last_message_at,
            "messages_received": telemetry.messages_received,
            "messages_rejected": telemetry.messages_rejected,
            "reconnect_attempts": telemetry.reconnect_attempts,
            "last_error": telemetry.last_error,
            "subscriptions": {
                "total_subscribed": sub_details["subscribed_count"],
                "active_instruments": sub_details["active_instruments"],
                "inactive_market_instruments": sub_details["inactive_market_instruments"],
                "capacity": self.config.max_subscriptions,
            },
        }

    def get_candles(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 50,
        include_in_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent candles (completed + in-progress)."""
        candles = self.candle_builder.get_recent_candles(
            symbol=symbol,
            interval=interval,
            limit=limit,
            include_in_progress=include_in_progress,
        )
        if not candles and interval.lower() == "1d":
            try:
                from ..services.market_data_service import MarketDataService
                mds = MarketDataService()
                df = mds.get_historical_dataframe(symbol)
                if df is not None and not df.empty:
                    records = df.tail(limit).to_dict(orient="records")
                    res = []
                    for r in records:
                        res.append({
                            "symbol": symbol,
                            "interval": "1d",
                            "start_time": str(r.get("Date") or r.get("start_time") or r.get("timestamp")),
                            "end_time": str(r.get("Date") or r.get("end_time") or r.get("timestamp")),
                            "open": float(r.get("Open", 0.0)),
                            "high": float(r.get("High", 0.0)),
                            "low": float(r.get("Low", 0.0)),
                            "close": float(r.get("Close", 0.0)),
                            "volume": float(r.get("Volume", 0.0)),
                            "source": "historical_csv",
                            "is_complete": True,
                        })
                    return res
            except Exception:
                pass
        return candles

    def get_in_progress_candle(self, symbol: str, interval: str = "5m") -> Optional[Dict[str, Any]]:
        """Retrieve active in-progress candle."""
        return self.candle_builder.get_in_progress_candle(symbol, interval=interval)

    def get_latest_completed_candle(self, symbol: str, interval: str = "5m") -> Optional[Dict[str, Any]]:
        """Retrieve latest closed candle."""
        return self.candle_builder.get_latest_completed_candle(symbol, interval=interval)

    def get_feature_readiness(
        self,
        symbol: str,
        interval: str = "5m",
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate PPO feature engineering readiness and timeframe compatibility."""
        clean_sym = str(symbol).strip().upper()
        candles = self.get_candles(clean_sym, interval=interval, limit=100, include_in_progress=False)

        # Fallback to historical daily candles if interval is daily and buffer is empty
        if len(candles) < 50 and interval.lower() == "1d":
            try:
                from ..services.market_data_service import MarketDataService
                mds = MarketDataService()
                df = mds.get_historical_dataframe(clean_sym)
                if df is not None and len(df) >= 50:
                    candles = df.tail(60).to_dict(orient="records")
                    for c in candles:
                        c["is_complete"] = True
            except Exception:
                pass

        return self.feature_pipeline.generate_live_observation(
            candles_history=candles,
            portfolio_state=portfolio_state,
            candle_interval=interval,
            symbol=clean_sym,
        )

    def get_dashboard_payload(
        self,
        symbol: str = "RELIANCE.NS",
        portfolio_name: str = "default",
        interval: str = "5m",
    ) -> Dict[str, Any]:
        """Aggregate all real-time market data, freshness, candles, and features for frontend rendering."""
        clean_sym = str(symbol).strip().upper()
        clean_inv = interval.lower() if interval in ("1m", "5m", "1d") else "5m"

        self.check_staleness()
        telemetry = self.provider.telemetry

        # 1. Fetch subscribed instruments and freshness telemetry
        subscribed = self.get_subscribed_symbols()
        if clean_sym not in subscribed and self.provider.status == ConnectionStatus.CONNECTED:
            self.provider.subscribe([clean_sym])
            subscribed = self.get_subscribed_symbols()

        quotes = self.get_all_quotes()
        freshness_map: Dict[str, Any] = {}
        quotes_summary: Dict[str, Any] = {}

        symbols_to_report = list(subscribed)
        if clean_sym not in symbols_to_report:
            symbols_to_report.append(clean_sym)

        for sym in symbols_to_report:
            f = self.get_symbol_freshness(sym)
            freshness_map[sym] = f
            q = quotes.get(sym) or self.get_latest_quote(sym)
            if q:
                quotes_summary[sym] = {
                    "symbol": sym,
                    "last_price": q.last_price,
                    "timestamp": q.timestamp,
                    "received_at": q.received_at,
                    "open": q.open_price,
                    "high": q.high_price,
                    "low": q.low_price,
                    "close": q.close_price,
                    "volume": q.volume,
                    "bid": q.bid_price,
                    "ask": q.ask_price,
                    "freshness_status": f["status"],
                }

        # 2. Candles for the selected symbol
        candles = self.get_candles(clean_sym, interval=clean_inv, limit=60, include_in_progress=True)
        in_prog = self.get_in_progress_candle(clean_sym, interval=clean_inv)
        latest_completed = self.get_latest_completed_candle(clean_sym, interval=clean_inv)

        # 3. PPO feature readiness
        features_status = self.get_feature_readiness(clean_sym, interval=clean_inv)

        return {
            "symbol": clean_sym,
            "interval": clean_inv,
            "portfolio_name": portfolio_name,
            "connection": {
                "status": telemetry.status.value,
                "provider": telemetry.provider_name,
                "is_stale": telemetry.is_stale,
                "last_heartbeat_at": telemetry.last_heartbeat_at,
                "last_message_at": telemetry.last_message_at,
                "messages_received": telemetry.messages_received,
                "messages_rejected": telemetry.messages_rejected,
            },
            "subscribed_symbols": subscribed,
            "quotes": quotes_summary,
            "freshness": freshness_map,
            "candles": candles,
            "in_progress_candle": in_prog,
            "latest_completed_candle": latest_completed,
            "features_readiness": features_status,
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    def sync_portfolio_subscriptions(
        self,
        portfolio_symbols: List[str],
        portfolio_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronize real-time subscriptions with a portfolio's stock selection bucket."""
        clean_target = {str(s).strip().upper() for s in portfolio_symbols if str(s).strip()}
        current_subs = set(self.provider.subscribed_symbols)

        to_add = sorted(list(clean_target - current_subs))
        to_remove = sorted(list(current_subs - clean_target))

        added_ok = True
        removed_ok = True

        if to_remove:
            removed_ok = self.provider.unsubscribe(to_remove)

        if to_add:
            added_ok = self.provider.subscribe(to_add)

        if portfolio_name:
            self._active_portfolio_name = portfolio_name

        return {
            "success": added_ok and removed_ok,
            "portfolio_name": self._active_portfolio_name,
            "subscribed_symbols": self.provider.subscribed_symbols,
            "added": to_add,
            "removed": to_remove,
            "count": len(self.provider.subscribed_symbols),
        }

    def check_staleness(self) -> bool:
        """Evaluate if connection is ostensibly open but updates have stopped arriving."""
        t = self.provider.telemetry
        if t.status != ConnectionStatus.CONNECTED:
            return False

        if t.subscribed_symbols_count == 0:
            return False

        now_dt = datetime.now(timezone.utc)
        reference_time_str = t.last_heartbeat_at or t.last_message_at

        if not reference_time_str:
            return False

        try:
            ref_dt = datetime.fromisoformat(reference_time_str.replace("Z", "+00:00"))
            elapsed = (now_dt - ref_dt).total_seconds()
            if elapsed > self.config.stale_timeout_seconds:
                t.is_stale = True
                t.status = ConnectionStatus.STALE
                t.last_error = f"Feed stale: No heartbeat received for {elapsed:.1f}s (threshold {self.config.stale_timeout_seconds}s)."
                self.provider.subscription_manager.evaluate_instrument_activity(is_connection_alive=False)
                return True
            else:
                t.is_stale = False
                self.provider.subscription_manager.evaluate_instrument_activity(is_connection_alive=True)
        except Exception:
            pass

        return False

    def _start_staleness_watchdog(self) -> None:
        """Start background daemon checking for feed staleness."""
        self._is_monitoring = True

        def watchdog_loop():
            while self._is_monitoring:
                time.sleep(2.0)
                if self._is_monitoring:
                    self.check_staleness()

        self._staleness_checker_thread = threading.Thread(
            target=watchdog_loop,
            daemon=True,
            name="RealtimeFeedWatchdogThread",
        )
        self._staleness_checker_thread.start()
