"""Real-Time Market Data Message Validator (Phase 11A).

Performs strict schema validation, price positivity checks, clock-skew checks,
staleness detection, and duplicate tick elimination on incoming market feeds.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

from .models import MarketQuote


class RealtimeDataValidator:
    """Validates real-time ticks, quotes, and market-data messages."""

    def __init__(
        self,
        max_clock_skew_seconds: float = 60.0,
        max_staleness_seconds: float = 300.0,
        out_of_order_tolerance_seconds: float = 0.0,
        supported_symbols: Optional[set] = None,
    ):
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.max_staleness_seconds = max_staleness_seconds
        self.out_of_order_tolerance_seconds = out_of_order_tolerance_seconds
        self.supported_symbols = supported_symbols or {
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
        }
        # Cache for duplicate and sequence detection: {symbol: (last_timestamp, last_price, last_seq)}
        self._last_seen: Dict[str, Tuple[datetime, float, Optional[int]]] = {}

    def parse_timestamp(self, ts_raw: Any) -> Optional[datetime]:
        """Parse raw timestamp (datetime, ISO string, epoch float/int) to UTC datetime."""
        if ts_raw is None:
            return None
        if isinstance(ts_raw, datetime):
            if ts_raw.tzinfo is None:
                return ts_raw.replace(tzinfo=timezone.utc)
            return ts_raw.astimezone(timezone.utc)
        if isinstance(ts_raw, (int, float)):
            try:
                # Distinguish milliseconds from seconds (epoch > 1e11 is ms)
                if ts_raw > 1e11:
                    ts_raw = ts_raw / 1000.0
                return datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            except Exception:
                return None
        if isinstance(ts_raw, str):
            ts_str = ts_raw.strip()
            # Try ISO formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ):
                try:
                    dt = datetime.strptime(ts_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                return None
        return None

    def validate_price(self, price: Any, field_name: str = "last_price") -> Tuple[bool, Optional[float], Optional[str]]:
        """Validate that price is a strictly positive, finite float."""
        if price is None:
            return False, None, f"Field '{field_name}' is missing."
        try:
            val = float(price)
        except (ValueError, TypeError):
            return False, None, f"Field '{field_name}' must be numeric (got {price})."

        if math.isnan(val) or math.isinf(val):
            return False, None, f"Field '{field_name}' cannot be NaN or Inf."
        if val <= 0.0:
            return False, None, f"Field '{field_name}' must be strictly positive (> 0, got {val})."

        return True, val, None

    def validate_message(
        self,
        raw_msg: Dict[str, Any],
        symbol_override: Optional[str] = None,
        source: str = "kite",
    ) -> Tuple[bool, Optional[MarketQuote], Optional[str]]:
        """Validate raw incoming tick dictionary and construct a normalized MarketQuote.
        
        Args:
            raw_msg: Raw tick dictionary from provider.
            symbol_override: Mapped symbol (if provider transmits raw integer instrument token).
            source: Source identifier string.
            
        Returns:
            Tuple of (is_valid, MarketQuote or None, error_message or None).
        """
        if not isinstance(raw_msg, dict):
            return False, None, f"Malformed message: expected dict, got {type(raw_msg).__name__}."

        # 1. Resolve & Validate Instrument Symbol
        symbol = symbol_override or raw_msg.get("symbol") or raw_msg.get("tradingsymbol")
        if not symbol or not str(symbol).strip():
            return False, None, "Message is missing a valid instrument identifier or symbol."
        clean_symbol = str(symbol).strip().upper()

        if self.supported_symbols and clean_symbol not in self.supported_symbols:
            return False, None, f"Symbol '{clean_symbol}' is not among supported instruments."

        # 2. Validate Last Traded Price (LTP)
        # Providers may name it last_price, ltp, price, or close
        raw_price = (
            raw_msg.get("last_price")
            if raw_msg.get("last_price") is not None
            else raw_msg.get("ltp", raw_msg.get("price"))
        )
        ok_price, last_price, price_err = self.validate_price(raw_price, "last_price")
        if not ok_price or last_price is None:
            return False, None, price_err

        # 3. Validate Event Timestamp
        now_utc = datetime.now(timezone.utc)
        raw_ts = (
            raw_msg.get("timestamp")
            or raw_msg.get("exchange_timestamp")
            or raw_msg.get("last_trade_time")
        )

        event_dt = self.parse_timestamp(raw_ts)
        if event_dt is None:
            # If provider message does not contain timestamp, reject or fallback to now
            return False, None, f"Message for '{clean_symbol}' lacks a valid parseable timestamp."

        # Future Clock-Skew Check
        skew_seconds = (event_dt - now_utc).total_seconds()
        if skew_seconds > self.max_clock_skew_seconds:
            return False, None, (
                f"Clock skew error: event timestamp {event_dt.isoformat()} is "
                f"{skew_seconds:.1f}s in the future (max allowed {self.max_clock_skew_seconds}s)."
            )

        # Staleness Check (Event timestamp is excessively old)
        age_seconds = (now_utc - event_dt).total_seconds()
        if age_seconds > self.max_staleness_seconds:
            return False, None, (
                f"Stale message: event timestamp {event_dt.isoformat()} is "
                f"{age_seconds:.1f}s old (max allowed {self.max_staleness_seconds}s)."
            )

        # 4. Duplicate Tick Detection
        seq = raw_msg.get("sequence_number") or raw_msg.get("seq")
        seq_num = int(seq) if seq is not None and str(seq).isdigit() else None

        last_seen = self._last_seen.get(clean_symbol)
        if last_seen:
            last_ts, last_p, last_seq = last_seen
            # Check if this tick is out-of-order or duplicate
            if event_dt < last_ts:
                lag = (last_ts - event_dt).total_seconds()
                if lag > self.out_of_order_tolerance_seconds:
                    return False, None, (
                        f"Out-of-order tick for '{clean_symbol}': event timestamp {event_dt.isoformat()} "
                        f"is {lag:.1f}s earlier than last processed tick {last_ts.isoformat()} (max tolerance {self.out_of_order_tolerance_seconds}s)."
                    )
            elif event_dt == last_ts and math.isclose(last_price, last_p, abs_tol=1e-5):
                if seq_num is None or seq_num == last_seq:
                    return False, None, f"Duplicate tick detected for '{clean_symbol}' at {event_dt.isoformat()}."

        # 5. Extract Optional OHLC & Depth Fields safely
        def safe_opt_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                return f if (not math.isnan(f) and not math.isinf(f) and f > 0) else None
            except Exception:
                return None

        def safe_opt_int(val: Any) -> Optional[int]:
            if val is None:
                return None
            try:
                i = int(val)
                return i if i >= 0 else None
            except Exception:
                return None

        ohlc = raw_msg.get("ohlc", {}) if isinstance(raw_msg.get("ohlc"), dict) else {}
        open_price = safe_opt_float(raw_msg.get("open") or ohlc.get("open"))
        high_price = safe_opt_float(raw_msg.get("high") or ohlc.get("high"))
        low_price = safe_opt_float(raw_msg.get("low") or ohlc.get("low"))
        close_price = safe_opt_float(raw_msg.get("close") or ohlc.get("close"))
        volume = safe_opt_int(raw_msg.get("volume"))

        depth = raw_msg.get("depth", {}) if isinstance(raw_msg.get("depth"), dict) else {}
        bids = depth.get("buy", [])
        asks = depth.get("sell", [])
        bid_price = safe_opt_float(bids[0].get("price")) if (bids and isinstance(bids, list) and isinstance(bids[0], dict)) else safe_opt_float(raw_msg.get("bid"))
        bid_qty = safe_opt_int(bids[0].get("quantity")) if (bids and isinstance(bids, list) and isinstance(bids[0], dict)) else safe_opt_int(raw_msg.get("bid_quantity"))
        ask_price = safe_opt_float(asks[0].get("price")) if (asks and isinstance(asks, list) and isinstance(asks[0], dict)) else safe_opt_float(raw_msg.get("ask"))
        ask_qty = safe_opt_int(asks[0].get("quantity")) if (asks and isinstance(asks, list) and isinstance(asks[0], dict)) else safe_opt_int(raw_msg.get("ask_quantity"))

        # Update cache watermark
        if not last_seen or event_dt >= last_seen[0]:
            self._last_seen[clean_symbol] = (event_dt, last_price, seq_num)

        quote = MarketQuote(
            symbol=clean_symbol,
            last_price=last_price,
            timestamp=event_dt.isoformat(),
            received_at=now_utc.isoformat(),
            volume=volume,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_quantity=bid_qty,
            ask_quantity=ask_qty,
            source=source,
            sequence_number=seq_num,
        )
        return True, quote, None
