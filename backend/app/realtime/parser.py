"""Zerodha Kite Connect WebSocket Protocol Parser (Phase 11B).

Decodes binary packets (LTP, Quote, Full modes) and text/JSON control messages
according to the official Kite Connect (KiteTicker) protocol specification.
Provides robust parsing with zero risk of unhandled exceptions crashing the service.
"""
from __future__ import annotations

import json
import logging
import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Packet lengths defined by KiteTicker protocol
PACKET_LEN_LTP = 8
PACKET_LEN_INDEX_QUOTE = 28
PACKET_LEN_INDEX_FULL = 32
PACKET_LEN_QUOTE = 44
PACKET_LEN_FULL = 184

NSE_EQUITY_DIVISOR = 100.0


class KiteProtocolParser:
    """Parser and encoder for KiteTicker WebSocket frames."""

    @staticmethod
    def parse_binary_message(data: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Parse binary message from KiteTicker into a list of raw tick dictionaries.
        
        Args:
            data: Raw binary byte frame from WebSocket.
            
        Returns:
            Tuple of (ticks_list, error_message). If parsing succeeds, error_message is None.
        """
        if not data or not isinstance(data, (bytes, bytearray)):
            return [], "Empty or non-binary data payload."

        if len(data) < 2:
            return [], f"Binary frame too short ({len(data)} bytes, expected at least 2)."

        try:
            num_packets = struct.unpack(">H", data[0:2])[0]
        except struct.error as exc:
            return [], f"Failed to read packet count header: {exc}"

        ticks: List[Dict[str, Any]] = []
        offset = 2

        for i in range(num_packets):
            if offset + 2 > len(data):
                err = f"Malformed binary frame: unexpected EOF before reading packet {i+1} length."
                logger.warning(err)
                return ticks, err

            try:
                packet_length = struct.unpack(">H", data[offset:offset + 2])[0]
            except struct.error as exc:
                return ticks, f"Failed to read packet length at offset {offset}: {exc}"

            offset += 2
            if offset + packet_length > len(data):
                err = f"Malformed binary frame: truncated packet {i+1} (expected {packet_length} bytes, have {len(data) - offset})."
                logger.warning(err)
                return ticks, err

            packet_data = data[offset:offset + packet_length]
            offset += packet_length

            try:
                tick = KiteProtocolParser._decode_packet(packet_data)
                if tick:
                    ticks.append(tick)
            except Exception as exc:
                logger.error("Error decoding individual packet: %s", exc)

        return ticks, None

    @staticmethod
    def _decode_packet(packet: bytes) -> Optional[Dict[str, Any]]:
        """Decode a single packet slice according to its length."""
        pkt_len = len(packet)
        if pkt_len < 4:
            return None

        # Every packet starts with uint32 instrument token
        instrument_token = struct.unpack(">I", packet[0:4])[0]

        now_iso = datetime.now(timezone.utc).isoformat()

        if pkt_len == PACKET_LEN_LTP:
            last_price = struct.unpack(">I", packet[4:8])[0] / NSE_EQUITY_DIVISOR
            return {
                "instrument_token": instrument_token,
                "mode": "ltp",
                "last_price": round(last_price, 2),
                "timestamp": now_iso,
            }

        elif pkt_len in (PACKET_LEN_INDEX_QUOTE, PACKET_LEN_INDEX_FULL):
            last_price = struct.unpack(">I", packet[4:8])[0] / NSE_EQUITY_DIVISOR
            high = struct.unpack(">I", packet[8:12])[0] / NSE_EQUITY_DIVISOR
            low = struct.unpack(">I", packet[12:16])[0] / NSE_EQUITY_DIVISOR
            open_p = struct.unpack(">I", packet[16:20])[0] / NSE_EQUITY_DIVISOR
            close_p = struct.unpack(">I", packet[20:24])[0] / NSE_EQUITY_DIVISOR
            return {
                "instrument_token": instrument_token,
                "mode": "index",
                "last_price": round(last_price, 2),
                "open_price": round(open_p, 2),
                "high_price": round(high, 2),
                "low_price": round(low, 2),
                "close_price": round(close_p, 2),
                "timestamp": now_iso,
            }

        elif pkt_len in (PACKET_LEN_QUOTE, PACKET_LEN_FULL):
            last_price = struct.unpack(">I", packet[4:8])[0] / NSE_EQUITY_DIVISOR
            last_traded_qty = struct.unpack(">I", packet[8:12])[0]
            avg_traded_price = struct.unpack(">I", packet[12:16])[0] / NSE_EQUITY_DIVISOR
            volume = struct.unpack(">I", packet[16:20])[0]
            buy_qty = struct.unpack(">I", packet[20:24])[0]
            sell_qty = struct.unpack(">I", packet[24:28])[0]
            open_p = struct.unpack(">I", packet[28:32])[0] / NSE_EQUITY_DIVISOR
            high_p = struct.unpack(">I", packet[32:36])[0] / NSE_EQUITY_DIVISOR
            low_p = struct.unpack(">I", packet[36:40])[0] / NSE_EQUITY_DIVISOR
            close_p = struct.unpack(">I", packet[40:44])[0] / NSE_EQUITY_DIVISOR

            tick: Dict[str, Any] = {
                "instrument_token": instrument_token,
                "mode": "quote" if pkt_len == PACKET_LEN_QUOTE else "full",
                "last_price": round(last_price, 2),
                "last_traded_quantity": last_traded_qty,
                "average_price": round(avg_traded_price, 2),
                "volume": volume,
                "buy_quantity": buy_qty,
                "sell_quantity": sell_qty,
                "open_price": round(open_p, 2),
                "high_price": round(high_p, 2),
                "low_price": round(low_p, 2),
                "close_price": round(close_p, 2),
                "timestamp": now_iso,
            }

            if pkt_len == PACKET_LEN_FULL:
                last_trade_time_raw = struct.unpack(">I", packet[44:48])[0]
                oi = struct.unpack(">I", packet[48:52])[0]
                oi_day_high = struct.unpack(">I", packet[52:56])[0]
                oi_day_low = struct.unpack(">I", packet[56:60])[0]
                exchange_timestamp_raw = struct.unpack(">I", packet[60:64])[0]

                # Convert epoch timestamp if valid (> 0)
                if exchange_timestamp_raw > 0:
                    try:
                        dt = datetime.fromtimestamp(exchange_timestamp_raw, tz=timezone.utc)
                        tick["timestamp"] = dt.isoformat()
                        tick["exchange_timestamp"] = dt.isoformat()
                    except Exception:
                        pass
                elif last_trade_time_raw > 0:
                    try:
                        dt = datetime.fromtimestamp(last_trade_time_raw, tz=timezone.utc)
                        tick["timestamp"] = dt.isoformat()
                    except Exception:
                        pass

                tick["oi"] = oi
                tick["oi_day_high"] = oi_day_high
                tick["oi_day_low"] = oi_day_low

                # Parse Market Depth: 10 entries * 12 bytes each (64 to 184)
                # First 5 entries = Bids, next 5 entries = Asks
                bids: List[Dict[str, Any]] = []
                asks: List[Dict[str, Any]] = []
                depth_offset = 64

                for d_idx in range(10):
                    entry = packet[depth_offset:depth_offset + 12]
                    depth_offset += 12
                    qty = struct.unpack(">I", entry[0:4])[0]
                    price = struct.unpack(">I", entry[4:8])[0] / NSE_EQUITY_DIVISOR
                    orders = struct.unpack(">H", entry[8:10])[0]

                    depth_item = {
                        "quantity": qty,
                        "price": round(price, 2),
                        "orders": orders,
                    }
                    if d_idx < 5:
                        bids.append(depth_item)
                    else:
                        asks.append(depth_item)

                tick["depth"] = {"buy": bids, "sell": asks}
                if bids and bids[0]["price"] > 0:
                    tick["bid"] = bids[0]["price"]
                    tick["bid_quantity"] = bids[0]["quantity"]
                if asks and asks[0]["price"] > 0:
                    tick["ask"] = asks[0]["price"]
                    tick["ask_quantity"] = asks[0]["quantity"]

            return tick

        return None

    @staticmethod
    def parse_text_message(text: str) -> Dict[str, Any]:
        """Parse text/JSON WebSocket control frames (acknowledgements, heartbeats, errors).
        
        Args:
            text: Text string received from WebSocket.
            
        Returns:
            Dict containing message type and parsed fields.
        """
        clean = (text or "").strip()
        if not clean:
            return {"type": "empty"}

        if clean.lower() == "ping":
            return {"type": "ping"}
        if clean.lower() == "pong":
            return {"type": "pong"}

        try:
            payload = json.loads(clean)
            if isinstance(payload, dict):
                # Standard KiteTicker JSON response format
                # e.g. {"type": "subscribe", "status": True, "data": [738561, 2953217]}
                # or {"type": "error", "message": "..."}
                msg_type = payload.get("type", "unknown")
                return {
                    "type": msg_type,
                    "status": payload.get("status", True),
                    "data": payload.get("data"),
                    "message": payload.get("message"),
                    "raw": payload,
                }
            return {"type": "json", "data": payload}
        except json.JSONDecodeError:
            return {"type": "text", "content": clean}

    @staticmethod
    def build_binary_packet(
        instrument_token: int,
        mode: str = "full",
        last_price: float = 1000.0,
        volume: int = 10000,
        open_price: float = 990.0,
        high_price: float = 1010.0,
        low_price: float = 985.0,
        close_price: float = 995.0,
        exchange_timestamp: Optional[int] = None,
        best_bid_price: float = 999.8,
        best_bid_qty: int = 100,
        best_ask_price: float = 1000.2,
        best_ask_qty: int = 150,
    ) -> bytes:
        """Helper to construct valid KiteTicker binary frames for automated testing.
        
        Supports LTP (8 bytes), Quote (44 bytes), and Full (184 bytes).
        Returns a complete binary message including the 2-byte packet count header.
        """
        token = int(instrument_token)
        lp_int = int(round(last_price * NSE_EQUITY_DIVISOR))

        if mode == "ltp":
            payload = struct.pack(">II", token, lp_int)
        elif mode == "quote":
            op_int = int(round(open_price * NSE_EQUITY_DIVISOR))
            hp_int = int(round(high_price * NSE_EQUITY_DIVISOR))
            low_int = int(round(low_price * NSE_EQUITY_DIVISOR))
            cp_int = int(round(close_price * NSE_EQUITY_DIVISOR))
            ltq = 10
            atp_int = lp_int
            buy_q = best_bid_qty
            sell_q = best_ask_qty
            payload = struct.pack(
                ">IIIIIIIIIII",
                token,
                lp_int,
                ltq,
                atp_int,
                int(volume),
                buy_q,
                sell_q,
                op_int,
                hp_int,
                low_int,
                cp_int,
            )
        else:  # full (184 bytes)
            op_int = int(round(open_price * NSE_EQUITY_DIVISOR))
            hp_int = int(round(high_price * NSE_EQUITY_DIVISOR))
            low_int = int(round(low_price * NSE_EQUITY_DIVISOR))
            cp_int = int(round(close_price * NSE_EQUITY_DIVISOR))
            ltq = 10
            atp_int = lp_int
            buy_q = best_bid_qty
            sell_q = best_ask_qty

            ts_epoch = exchange_timestamp if exchange_timestamp is not None else int(datetime.now(timezone.utc).timestamp())
            quote_header = struct.pack(
                ">IIIIIIIIIII",
                token,
                lp_int,
                ltq,
                atp_int,
                int(volume),
                buy_q,
                sell_q,
                op_int,
                hp_int,
                low_int,
                cp_int,
            )
            extra_header = struct.pack(
                ">IIIII",
                ts_epoch,  # last_trade_time
                0,         # oi
                0,         # oi_day_high
                0,         # oi_day_low
                ts_epoch,  # exchange_timestamp
            )

            # 10 depth entries of 12 bytes each (64 to 184)
            depth_entries = bytearray()
            # 5 bids
            for b_idx in range(5):
                b_price = int(round((best_bid_price - b_idx * 0.1) * NSE_EQUITY_DIVISOR))
                b_qty = best_bid_qty + b_idx * 10
                depth_entries.extend(struct.pack(">IIHH", b_qty, b_price, 1, 0))
            # 5 asks
            for a_idx in range(5):
                a_price = int(round((best_ask_price + a_idx * 0.1) * NSE_EQUITY_DIVISOR))
                a_qty = best_ask_qty + a_idx * 10
                depth_entries.extend(struct.pack(">IIHH", a_qty, a_price, 1, 0))

            payload = quote_header + extra_header + bytes(depth_entries)

        # Wrap in message frame: 2 bytes packet count (1) + 2 bytes packet length + payload
        frame = struct.pack(">HH", 1, len(payload)) + payload
        return frame
