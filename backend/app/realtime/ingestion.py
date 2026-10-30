"""Live Market Data Ingestion and Validation Service (Phase 11C).

Ingests streaming ticks from real-time providers, enforces strict validation,
detects out-of-order and duplicate messages, separates event time from local
processing time, tracks instrument freshness, and forwards validated updates
to downstream candle builders without blocking the WebSocket loop.
"""
from __future__ import annotations

import collections
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .models import MarketQuote
from .validator import RealtimeDataValidator

logger = logging.getLogger(__name__)


class LiveMarketDataIngestionService:
    """Thread-safe ingestion service for validating, deduplicating, and tracking live market data."""

    def __init__(
        self,
        validator: Optional[RealtimeDataValidator] = None,
        max_recent_ticks: int = 500,
        freshness_threshold_seconds: float = 15.0,
        aging_threshold_seconds: float = 30.0,
        out_of_order_tolerance_seconds: float = 5.0,
    ):
        self.max_recent_ticks = max_recent_ticks
        self.freshness_threshold_seconds = freshness_threshold_seconds
        self.aging_threshold_seconds = aging_threshold_seconds
        self.out_of_order_tolerance_seconds = out_of_order_tolerance_seconds
        self.validator = validator or RealtimeDataValidator(out_of_order_tolerance_seconds=out_of_order_tolerance_seconds)

        self._lock = threading.Lock()
        # Latest validated quote per symbol
        self._latest_quotes: Dict[str, MarketQuote] = {}
        # Bounded sliding buffer of recent ticks per symbol
        self._recent_ticks: Dict[str, Deque[MarketQuote]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_recent_ticks)
        )
        # Seen tick signatures for duplicate detection: set of (symbol, event_ts, price, volume)
        self._seen_signatures: Deque[str] = collections.deque(maxlen=2000)
        self._seen_set: Set[str] = set()

        # Watermark timestamps per symbol to detect out-of-order arrivals
        self._latest_event_ts: Dict[str, datetime] = {}

        # Downstream update consumers (e.g. CandleBuilderService)
        self._consumers: List[Callable[[MarketQuote], None]] = []

        # Telemetry
        self._total_ingested = 0
        self._total_accepted = 0
        self._total_duplicates = 0
        self._total_out_of_order = 0
        self._total_rejected = 0

    def register_consumer(self, consumer: Callable[[MarketQuote], None]) -> None:
        """Register a downstream consumer callback invoked on every accepted quote."""
        with self._lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def unregister_consumer(self, consumer: Callable[[MarketQuote], None]) -> None:
        """Unregister a downstream consumer."""
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)

    def ingest_quote(self, quote: MarketQuote) -> Tuple[bool, Optional[str]]:
        """Ingest an already-structured MarketQuote directly into the ingestion processing pipeline.
        
        Returns:
            Tuple of (accepted: bool, rejection_reason: Optional[str]).
        """
        if not isinstance(quote, MarketQuote):
            return False, "Expected MarketQuote instance."
        return self._process_validated_quote(quote)

    def ingest_raw_message(
        self,
        raw_msg: Dict[str, Any],
        source: str = "kite",
        symbol_override: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate, deduplicate, process, and forward a raw provider tick message.
        
        Returns:
            Tuple of (accepted: bool, reason: Optional[str]).
        """
        with self._lock:
            self._total_ingested += 1

            # 1. Validation via RealtimeDataValidator
            is_valid, validated_quote, err = self.validator.validate_message(
                raw_msg=raw_msg,
                symbol_override=symbol_override,
                source=source,
            )
            if not is_valid or validated_quote is None:
                if err and "duplicate" in err.lower():
                    self._total_duplicates += 1
                else:
                    self._total_rejected += 1
                return False, f"Validation failed: {err}"

        return self._process_validated_quote(validated_quote)

    def _process_validated_quote(self, validated_quote: MarketQuote) -> Tuple[bool, Optional[str]]:
        """Deduplicate, check ordering, cache, and dispatch a validated MarketQuote."""
        with self._lock:
            sym = validated_quote.symbol
            price = validated_quote.last_price
            ts_str = validated_quote.timestamp

            # 2. Duplicate Detection
            sig = f"{sym}|{ts_str}|{price:.2f}|{validated_quote.volume or 0}"
            if sig in self._seen_set:
                self._total_duplicates += 1
                return False, "Duplicate tick discarded."

            # Record signature in bounded window
            if len(self._seen_signatures) >= self._seen_signatures.maxlen:
                oldest = self._seen_signatures.popleft()
                self._seen_set.discard(oldest)
            self._seen_signatures.append(sig)
            self._seen_set.add(sig)

            # 3. Out-of-Order Detection & Policy
            is_out_of_order = False
            try:
                event_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                last_dt = self._latest_event_ts.get(sym)

                if last_dt is not None:
                    if event_dt < last_dt:
                        diff_sec = (last_dt - event_dt).total_seconds()
                        if diff_sec <= self.out_of_order_tolerance_seconds:
                            # Acceptable late-arriving update within tolerance
                            is_out_of_order = True
                            self._total_out_of_order += 1
                            logger.debug(
                                "Late tick accepted for %s (lag: %.2fs within %.2fs tolerance)",
                                sym, diff_sec, self.out_of_order_tolerance_seconds
                            )
                        else:
                            self._total_rejected += 1
                            return False, f"Tick excessively out of order ({diff_sec:.1f}s older than latest seen)."
                    else:
                        # Advance event timestamp watermark
                        self._latest_event_ts[sym] = event_dt
                else:
                    self._latest_event_ts[sym] = event_dt
            except Exception as exc:
                logger.warning("Error evaluating tick timestamp ordering: %s", exc)

            # 4. Cache latest validated quote and update ring buffer
            self._latest_quotes[sym] = validated_quote
            self._recent_ticks[sym].append(validated_quote)
            self._total_accepted += 1

            # 5. Dispatch to downstream consumers outside any heavy locks
            consumers_snapshot = list(self._consumers)

        for consumer in consumers_snapshot:
            try:
                consumer(validated_quote)
            except Exception as exc:
                logger.error("Error in ingestion consumer: %s", exc)

        return True, None

    def get_latest_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Retrieve the freshest validated quote for symbol."""
        with self._lock:
            return self._latest_quotes.get(str(symbol).strip().upper())

    def get_all_latest_quotes(self) -> Dict[str, MarketQuote]:
        """Retrieve all current latest quotes."""
        with self._lock:
            return dict(self._latest_quotes)

    def get_recent_ticks(self, symbol: str, limit: int = 50) -> List[MarketQuote]:
        """Retrieve bounded history of recent ticks for an instrument."""
        with self._lock:
            sym = str(symbol).strip().upper()
            ticks = list(self._recent_ticks.get(sym, []))
            return ticks[-limit:] if limit > 0 else ticks

    def get_symbol_freshness(self, symbol: str) -> Dict[str, Any]:
        """Assess data freshness and processing latency for an instrument.
        
        Returns:
            Dict containing freshness status (FRESH, AGING, STALE, OFFLINE),
            event timestamp, received timestamp, and elapsed latency in ms.
        """
        with self._lock:
            sym = str(symbol).strip().upper()
            quote = self._latest_quotes.get(sym)

            if quote is None:
                return {
                    "symbol": sym,
                    "status": "OFFLINE",
                    "freshness": "No updates received",
                    "last_price": None,
                    "event_timestamp": None,
                    "received_at": None,
                    "latency_ms": None,
                    "elapsed_seconds": None,
                }

            now = datetime.now(timezone.utc)
            try:
                recv_dt = datetime.fromisoformat(quote.received_at.replace("Z", "+00:00"))
                elapsed = (now - recv_dt).total_seconds()
            except Exception:
                elapsed = 999.0

            try:
                event_dt = datetime.fromisoformat(quote.timestamp.replace("Z", "+00:00"))
                recv_dt = datetime.fromisoformat(quote.received_at.replace("Z", "+00:00"))
                latency_ms = max(0, int((recv_dt - event_dt).total_seconds() * 1000.0))
            except Exception:
                latency_ms = None

            if elapsed <= self.freshness_threshold_seconds:
                status = "FRESH"
            elif elapsed <= self.aging_threshold_seconds:
                status = "AGING"
            else:
                status = "STALE"

            return {
                "symbol": sym,
                "status": status,
                "last_price": quote.last_price,
                "event_timestamp": quote.timestamp,
                "received_at": quote.received_at,
                "latency_ms": latency_ms,
                "elapsed_seconds": round(elapsed, 1),
                "source": quote.source,
            }

    def get_telemetry(self) -> Dict[str, Any]:
        """Retrieve ingestion service diagnostic telemetry."""
        with self._lock:
            return {
                "total_ingested": self._total_ingested,
                "total_accepted": self._total_accepted,
                "total_duplicates": self._total_duplicates,
                "total_out_of_order": self._total_out_of_order,
                "total_rejected": self._total_rejected,
                "symbols_tracked": len(self._latest_quotes),
            }
