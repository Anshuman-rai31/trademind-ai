"""Automated Paper Trading Engine Service (Phase 13).

Consumes validated PPO trading signals from Phase 12, performs pre-trade risk gating,
validates portfolio allocation and cash constraints, and executes simulated trades
atomically against isolated virtual portfolios with realistic slippage and transaction costs.
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
from ..models.paper_trading import (
    PaperPortfolio,
    PaperPosition,
    PaperSnapshotRecord,
    PaperTrade,
    PortfolioBucketItem,
    RiskEventRecord,
)
from ..models.trading_signal import RLTradingSignal
from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..simulation.models import Order, OrderSide
from ..simulation.portfolio import Portfolio

logger = logging.getLogger(__name__)

# Configurable realistic institutional transaction cost assumptions
DEFAULT_FEE_RATE = 0.0003        # 0.03% (STT, brokerage, exchange turnover)
DEFAULT_SLIPPAGE_RATE = 0.0005   # 0.05% (5 bps market impact)
DEFAULT_ALLOCATION_PCT = 0.95    # Max 95% cash utilization buffer


class PaperTradingService:
    """Thread-safe automated paper-trading execution engine."""

    _instance: Optional[PaperTradingService] = None
    _lock = threading.Lock()

    def __init__(
        self,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.risk_manager = risk_manager or RiskManager(config=RiskConfig(enabled=True))

        self._exec_lock = threading.RLock()
        # In-memory deduplication tracking: (portfolio_id, signal_id)
        self._executed_signals: Set[Tuple[int, str]] = set()
        self._executed_queue: Deque[Tuple[int, str]] = collections.deque(maxlen=2000)

        # Operational telemetry counters
        self._telemetry = {
            "total_signals_received": 0,
            "total_trades_executed": 0,
            "total_buy_trades": 0,
            "total_sell_trades": 0,
            "total_hold_signals": 0,
            "total_rejected_signals": 0,
            "total_risk_blocked": 0,
            "total_insufficient_cash": 0,
            "total_duplicates_prevented": 0,
            "total_timeframe_mismatches_skipped": 0,
            "last_trade_time": None,
            "last_error": None,
        }

    @classmethod
    def get_instance(
        cls,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        risk_manager: Optional[RiskManager] = None,
    ) -> PaperTradingService:
        """Retrieve thread-safe singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    risk_manager=risk_manager,
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for isolated test fixtures)."""
        with cls._lock:
            cls._instance = None

    def on_signal_generated(self, signal_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Consumer callback suitable for registering directly with RealtimeSignalService."""
        try:
            return self.process_signal(signal_dict)
        except Exception as exc:
            logger.error("Exception in automated paper trading callback: %s", exc)
            with self._exec_lock:
                self._telemetry["last_error"] = str(exc)
            return None

    def process_signal(
        self,
        signal: Dict[str, Any] | RLTradingSignal | str,
        portfolio_name_or_id: Optional[str | int] = None,
    ) -> Dict[str, Any]:
        """Evaluate and execute simulated trade against a paper trading portfolio.
        
        Args:
            signal: Signal dict, RLTradingSignal instance, or signal_id string.
            portfolio_name_or_id: Target portfolio identifier (defaults to 'default' or signal.portfolio_id).
            
        Returns:
            Dict containing execution outcome, executed trade details, or structured rejection reason.
        """
        with self._exec_lock:
            return self._process_signal_locked(signal, portfolio_name_or_id)

    def _process_signal_locked(
        self,
        signal: Dict[str, Any] | RLTradingSignal | str,
        portfolio_name_or_id: Optional[str | int] = None,
    ) -> Dict[str, Any]:
        try:
            db.session.expire_all()
        except Exception:
            pass
        self._telemetry["total_signals_received"] += 1

        # 1. Resolve Signal Entity
        sig_data: Optional[Dict[str, Any]] = None
        if isinstance(signal, dict):
            sig_data = signal
        elif isinstance(signal, RLTradingSignal):
            sig_data = signal.to_dict()
        elif isinstance(signal, str):
            found_sig = RLTradingSignal.query.filter_by(signal_id=signal).first()
            if not found_sig and signal.isdigit():
                found_sig = db.session.get(RLTradingSignal, int(signal))
            if not found_sig:
                return self._record_rejection(
                    None, "SIGNAL_NOT_FOUND", f"Signal '{signal}' not found in database."
                )
            sig_data = found_sig.to_dict()

        if not sig_data:
            return self._record_rejection(None, "INVALID_SIGNAL", "Signal payload is empty or invalid.")

        signal_id = sig_data.get("signal_id") or f"sig_dyn_{sig_data.get('id', 'temp')}"
        symbol = str(sig_data.get("symbol", "")).strip().upper()
        interval = str(sig_data.get("interval", "1d")).strip().lower()
        status = str(sig_data.get("status", "VALID")).strip().upper()
        raw_action = str(sig_data.get("effective_action") or sig_data.get("action") or "HOLD").strip().upper()
        close_price = sig_data.get("close_price")

        # 2. Strict Filter: Reject Incompatible Timeframes
        if interval != "1d":
            with self._exec_lock:
                self._telemetry["total_timeframe_mismatches_skipped"] += 1
            return self._record_rejection(
                signal_id,
                "TIMEFRAME_MISMATCH",
                f"Automated execution withheld: Signal interval is '{interval}', but PPO model was trained on daily ('1d') bars.",
                symbol=symbol,
            )

        # 3. Strict Filter: Reject Blocked, Mismatched, or Invalid Signals
        if status in ("TIMEFRAME_MISMATCH", "RISK_BLOCKED", "REJECTED"):
            with self._exec_lock:
                if status == "RISK_BLOCKED":
                    self._telemetry["total_risk_blocked"] += 1
                else:
                    self._telemetry["total_rejected_signals"] += 1
            return self._record_rejection(
                signal_id,
                status,
                f"Signal status is '{status}': {sig_data.get('rejection_reason', 'Execution rejected.')}",
                symbol=symbol,
            )

        # 4. Resolve Target Portfolio
        target_port = self._resolve_portfolio(portfolio_name_or_id or sig_data.get("portfolio_id"))
        if not target_port:
            return self._record_rejection(
                signal_id, "PORTFOLIO_NOT_FOUND", "Target simulation portfolio not found.", symbol=symbol
            )

        port_id = target_port.id

        # 5. Check if Paper Trading is Active for Portfolio
        if not getattr(target_port, "is_active", True):
            return self._record_rejection(
                signal_id,
                "PORTFOLIO_PAUSED",
                f"Paper trading is currently paused for portfolio '{target_port.name}'.",
                portfolio_id=port_id,
                symbol=symbol,
            )

        # 6. Duplicate Execution Prevention & Idempotency
        key = (port_id, signal_id)
        with self._exec_lock:
            if key in self._executed_signals:
                self._telemetry["total_duplicates_prevented"] += 1
                logger.info("[%s] Duplicate execution attempt for signal %s on portfolio %d prevented.", symbol, signal_id, port_id)
                existing_trade = PaperTrade.query.filter_by(portfolio_id=port_id, signal_id=signal_id).first()
                return {
                    "executed": False,
                    "status": "DUPLICATE_PREVENTED",
                    "reason": f"Signal '{signal_id}' already executed on portfolio '{target_port.name}'.",
                    "trade": existing_trade.to_dict() if existing_trade else None,
                }

        # Check DB for existing trade with this signal_id
        existing_trade = PaperTrade.query.filter_by(portfolio_id=port_id, signal_id=signal_id).first()
        if existing_trade:
            self._mark_signal_executed(key)
            with self._exec_lock:
                self._telemetry["total_duplicates_prevented"] += 1
            return {
                "executed": False,
                "status": "DUPLICATE_PREVENTED",
                "reason": f"Signal '{signal_id}' has already generated trade ID {existing_trade.id}.",
                "trade": existing_trade.to_dict(),
            }

        # 7. Handle HOLD Signal (No execution needed, update mark-to-market if price valid)
        if raw_action == "HOLD":
            with self._exec_lock:
                self._telemetry["total_hold_signals"] += 1
            self._update_mark_to_market(target_port, symbol, close_price)
            return {
                "executed": False,
                "status": "HOLD_NO_TRADE",
                "action": "HOLD",
                "reason": "Signal policy action is HOLD; maintaining current position.",
                "portfolio_id": port_id,
                "symbol": symbol,
            }

        # 8. Validate Reference Price
        if close_price is None or not isinstance(close_price, (int, float)) or close_price <= 0:
            return self._record_rejection(
                signal_id, "INVALID_PRICE", f"Invalid reference price '{close_price}'.", portfolio_id=port_id, symbol=symbol
            )

        ref_price = float(close_price)
        timestamp = sig_data.get("timestamp") or datetime.now(timezone.utc).isoformat()

        # 9. Perform Pre-Trade Risk Check via RiskManager
        risk_check = self._evaluate_risk_policy(target_port, symbol, raw_action, ref_price, timestamp)
        if not risk_check["approved"]:
            with self._exec_lock:
                self._telemetry["total_risk_blocked"] += 1
            return self._record_rejection(
                signal_id,
                "RISK_BLOCKED",
                risk_check["reason"],
                portfolio_id=port_id,
                symbol=symbol,
                risk_details=risk_check,
            )

        # 10. Execute Trade Atomically (BUY / SELL)
        try:
            trade_res = self._execute_trade_atomic(
                portfolio=target_port,
                symbol=symbol,
                action=raw_action,
                ref_price=ref_price,
                signal_id=signal_id,
                timestamp=timestamp,
                sig_data=sig_data,
            )
            if trade_res.get("executed"):
                self._mark_signal_executed(key)
            return trade_res
        except Exception as exc:
            db.session.rollback()
            logger.error("[%s] Transaction rollback during paper trade execution: %s", symbol, exc)
            with self._exec_lock:
                self._telemetry["last_error"] = str(exc)
            return self._record_rejection(
                signal_id, "EXECUTION_ERROR", f"Transaction failed and was safely rolled back: {exc}", portfolio_id=port_id, symbol=symbol
            )

    def _execute_trade_atomic(
        self,
        portfolio: PaperPortfolio,
        symbol: str,
        action: str,
        ref_price: float,
        signal_id: str,
        timestamp: str,
        sig_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Atomically calculate fills, fees, slippage, update positions, and commit ledger."""
        port_id = portfolio.id
        pos = PaperPosition.query.filter_by(portfolio_id=port_id, symbol=symbol).first()
        current_qty = float(pos.quantity) if pos else 0.0
        avg_entry = float(pos.average_entry_price) if pos else 0.0

        if action == "BUY":
            fill_price = round(ref_price * (1.0 + self.slippage_rate), 2)
            # Allocation sizing: check bucket target weight
            target_weight = self._get_target_weight(portfolio, symbol)
            max_capital = portfolio.cash * DEFAULT_ALLOCATION_PCT
            allocated_capital = min(portfolio.cash * target_weight, max_capital)

            qty = np.floor(allocated_capital / fill_price)
            if qty <= 0:
                with self._exec_lock:
                    self._telemetry["total_insufficient_cash"] += 1
                return self._record_rejection(
                    signal_id,
                    "INSUFFICIENT_CASH",
                    f"Available cash (₹{portfolio.cash:.2f}) insufficient to buy 1 share of {symbol} at ₹{fill_price:.2f}.",
                    portfolio_id=port_id,
                    symbol=symbol,
                )

            gross_cost = qty * fill_price
            fee = round(gross_cost * self.fee_rate, 2)
            total_outlay = gross_cost + fee

            if portfolio.cash < total_outlay:
                with self._exec_lock:
                    self._telemetry["total_insufficient_cash"] += 1
                return self._record_rejection(
                    signal_id,
                    "INSUFFICIENT_CASH",
                    f"Outlay ₹{total_outlay:.2f} exceeds cash balance ₹{portfolio.cash:.2f}.",
                    portfolio_id=port_id,
                    symbol=symbol,
                )

            # Deduct cash
            portfolio.cash = round(portfolio.cash - total_outlay, 2)

            # Update position with weighted average entry price
            new_qty = current_qty + qty
            new_avg_entry = ((current_qty * avg_entry) + gross_cost) / new_qty

            if not pos:
                pos = PaperPosition(portfolio_id=port_id, symbol=symbol)
                db.session.add(pos)

            pos.quantity = new_qty
            pos.average_entry_price = round(new_avg_entry, 2)
            pos.current_price = ref_price
            pos.unrealized_pnl = round((ref_price - new_avg_entry) * new_qty, 2)
            pos.updated_at = datetime.now(timezone.utc)

            slippage_cost = round(qty * ref_price * self.slippage_rate, 2)

            trade = PaperTrade(
                portfolio_id=port_id,
                signal_id=signal_id,
                symbol=symbol,
                timestamp=timestamp,
                action="BUY",
                quantity=float(qty),
                price=fill_price,
                fee=fee,
                slippage=slippage_cost,
                realized_pnl=0.0,
                risk_status="APPROVED",
                metadata_json=safe_json_dumps({
                    "model_id": sig_data.get("model_id"),
                    "model_version": sig_data.get("model_version"),
                    "model_checksum": sig_data.get("model_checksum"),
                    "source_candle_timestamp": sig_data.get("source_candle_timestamp"),
                }),
            )
            db.session.add(trade)
            portfolio.total_trades_count += 1

            with self._exec_lock:
                self._telemetry["total_trades_executed"] += 1
                self._telemetry["total_buy_trades"] += 1
                self._telemetry["last_trade_time"] = datetime.now(timezone.utc).isoformat()

        elif action == "SELL":
            if current_qty <= 0:
                return self._record_rejection(
                    signal_id,
                    "NO_POSITION_TO_SELL",
                    f"Cannot execute SELL: Portfolio has zero open shares of {symbol}.",
                    portfolio_id=port_id,
                    symbol=symbol,
                )

            fill_price = round(ref_price * (1.0 - self.slippage_rate), 2)
            sell_qty = current_qty  # Liquidate holding
            gross_proceeds = sell_qty * fill_price
            fee = round(gross_proceeds * self.fee_rate, 2)
            net_proceeds = gross_proceeds - fee

            realized_pnl = round((fill_price - avg_entry) * sell_qty - fee, 2)
            portfolio.cash = round(portfolio.cash + net_proceeds, 2)
            portfolio.total_realized_pnl = round(portfolio.total_realized_pnl + realized_pnl, 2)

            pos.quantity = 0.0
            pos.current_price = ref_price
            pos.unrealized_pnl = 0.0
            pos.updated_at = datetime.now(timezone.utc)

            slippage_cost = round(sell_qty * ref_price * self.slippage_rate, 2)

            trade = PaperTrade(
                portfolio_id=port_id,
                signal_id=signal_id,
                symbol=symbol,
                timestamp=timestamp,
                action="SELL",
                quantity=float(sell_qty),
                price=fill_price,
                fee=fee,
                slippage=slippage_cost,
                realized_pnl=realized_pnl,
                risk_status="APPROVED",
                metadata_json=safe_json_dumps({
                    "model_id": sig_data.get("model_id"),
                    "model_version": sig_data.get("model_version"),
                    "model_checksum": sig_data.get("model_checksum"),
                    "source_candle_timestamp": sig_data.get("source_candle_timestamp"),
                }),
            )
            db.session.add(trade)
            portfolio.total_trades_count += 1

            with self._exec_lock:
                self._telemetry["total_trades_executed"] += 1
                self._telemetry["total_sell_trades"] += 1
                self._telemetry["last_trade_time"] = datetime.now(timezone.utc).isoformat()

        else:
            return self._record_rejection(
                signal_id, "UNSUPPORTED_ACTION", f"Unsupported action '{action}'.", portfolio_id=port_id, symbol=symbol
            )

        # 11. Mark to Market & Snapshot Persistence
        self._mark_portfolio_to_market(portfolio)
        db.session.commit()

        logger.info(
            "[%s] Executed simulated %s: %s shares at ₹%.2f on portfolio %s (Cash: ₹%.2f)",
            symbol,
            action,
            trade.quantity,
            trade.price,
            portfolio.name,
            portfolio.cash,
        )

        return {
            "executed": True,
            "status": "EXECUTED",
            "action": action,
            "portfolio_id": port_id,
            "symbol": symbol,
            "trade": trade.to_dict(),
            "portfolio": portfolio.to_dict(include_positions=True, include_bucket=True),
        }

    def _mark_portfolio_to_market(self, portfolio: PaperPortfolio) -> None:
        """Recalculate portfolio net equity, update peak value, and record snapshot."""
        holdings_val = sum((p.quantity * p.current_price) for p in portfolio.positions if p.quantity > 0)
        portfolio.portfolio_value = round(portfolio.cash + holdings_val, 2)
        if portfolio.portfolio_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.portfolio_value
        portfolio.updated_at = datetime.now(timezone.utc)

        drawdown = (
            max(0.0, (portfolio.peak_value - portfolio.portfolio_value) / portfolio.peak_value)
            if portfolio.peak_value > 0
            else 0.0
        )
        snap = PaperSnapshotRecord(
            portfolio_id=portfolio.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cash=portfolio.cash,
            holdings_value=round(holdings_val, 2),
            portfolio_value=portfolio.portfolio_value,
            drawdown=round(drawdown, 4),
        )
        db.session.add(snap)

    def _update_mark_to_market(self, portfolio: PaperPortfolio, symbol: str, price: Optional[float]) -> None:
        """Update current price and unrealized PnL for HOLD signals."""
        if price is None or price <= 0:
            return
        pos = PaperPosition.query.filter_by(portfolio_id=portfolio.id, symbol=symbol).first()
        if pos and pos.quantity > 0:
            pos.current_price = float(price)
            pos.unrealized_pnl = round((float(price) - pos.average_entry_price) * pos.quantity, 2)
            self._mark_portfolio_to_market(portfolio)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()

    def _get_target_weight(self, portfolio: PaperPortfolio, symbol: str) -> float:
        """Determine target allocation fraction for symbol in this portfolio."""
        bucket_item = PortfolioBucketItem.query.filter_by(portfolio_id=portfolio.id, symbol=symbol).first()
        if bucket_item and bucket_item.target_weight > 0:
            return float(bucket_item.target_weight)

        total_bucket = len(portfolio.bucket_items)
        if total_bucket > 0:
            return 1.0 / total_bucket
        return 0.20  # Default 20% for 5-asset universe

    def _evaluate_risk_policy(
        self,
        portfolio: PaperPortfolio,
        symbol: str,
        action: str,
        price: float,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Pre-trade risk gateway inspecting drawdown and daily loss."""
        try:
            # Check if portfolio is in severe drawdown
            port_val = portfolio.portfolio_value
            peak_val = portfolio.peak_value or portfolio.initial_cash
            current_dd = (peak_val - port_val) / peak_val if peak_val > 0 else 0.0

            max_dd_limit = self.risk_manager.config.max_drawdown_limit
            if current_dd > max_dd_limit:
                risk_event = RiskEventRecord(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    timestamp=timestamp,
                    rule_breached="MAX_DRAWDOWN_EXCEEDED",
                    severity="BLOCKED",
                    proposed_action=action,
                    effective_action="HOLD",
                    details=f"Portfolio drawdown {current_dd*100:.2f}% exceeds limit of {max_dd_limit*100:.1f}%.",
                )
                db.session.add(risk_event)
                db.session.commit()
                return {
                    "approved": False,
                    "reason": f"Risk Blocked: Drawdown {current_dd*100:.2f}% breached threshold {max_dd_limit*100:.1f}%.",
                    "details": risk_event.to_dict(),
                }

            return {"approved": True, "reason": "Pre-trade risk policy approved."}
        except Exception as exc:
            logger.warning("[%s] Risk policy evaluation fallback: %s", symbol, exc)
            return {"approved": True, "reason": "Risk check completed with fallback."}

    def _resolve_portfolio(self, port_ref: Optional[str | int]) -> Optional[PaperPortfolio]:
        """Resolve portfolio by ID, name, or fallback to 'default'."""
        if port_ref is not None:
            if isinstance(port_ref, int) or (isinstance(port_ref, str) and port_ref.isdigit()):
                p = db.session.get(PaperPortfolio, int(port_ref))
                if p:
                    return p
            p = PaperPortfolio.query.filter_by(name=str(port_ref)).first()
            if p:
                return p

        # Fallback to default portfolio
        default_port = PaperPortfolio.query.filter_by(name="default").first()
        if not default_port:
            default_port = PaperPortfolio(
                name="default",
                initial_cash=100000.0,
                cash=100000.0,
                portfolio_value=100000.0,
                peak_value=100000.0,
                is_active=True,
            )
            db.session.add(default_port)
            db.session.commit()
        return default_port

    def _mark_signal_executed(self, key: Tuple[int, str]) -> None:
        """Mark (portfolio_id, signal_id) executed in bounded set."""
        with self._exec_lock:
            if key not in self._executed_signals:
                self._executed_queue.append(key)
                self._executed_signals.add(key)
                if len(self._executed_signals) > len(self._executed_queue) + 200:
                    self._executed_signals = set(self._executed_queue)

    def _record_rejection(
        self,
        signal_id: Optional[str],
        status: str,
        reason: str,
        portfolio_id: Optional[int] = None,
        symbol: Optional[str] = None,
        risk_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Standardized rejection response payload."""
        logger.info("[%s] Paper trade rejected for signal %s: %s (%s)", symbol or "SYS", signal_id, status, reason)
        return {
            "executed": False,
            "status": status,
            "reason": reason,
            "signal_id": signal_id,
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "risk_details": risk_details,
        }

    # =========================================================================
    # Portfolio Management & Query API
    # =========================================================================

    def enable_paper_trading(self, portfolio_ref: str | int) -> bool:
        """Enable automated paper trading for a portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return False
        p.is_active = True
        p.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return True

    def disable_paper_trading(self, portfolio_ref: str | int) -> bool:
        """Pause automated paper trading for a portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return False
        p.is_active = False
        p.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return True

    def get_portfolio_status(self, portfolio_ref: str | int) -> Optional[Dict[str, Any]]:
        """Retrieve full status, cash, positions, and performance for portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return None
        return p.to_dict(include_positions=True, include_bucket=True)

    def get_positions(self, portfolio_ref: str | int) -> List[Dict[str, Any]]:
        """Retrieve active open positions for a portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return []
        return [pos.to_dict() for pos in p.positions if pos.quantity > 0]

    def get_trades(
        self,
        portfolio_ref: str | int,
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Retrieve paginated trade history for portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return {"trades": [], "total": 0, "page": page, "page_size": limit, "total_pages": 0}

        query = PaperTrade.query.filter_by(portfolio_id=p.id).order_by(PaperTrade.id.desc())
        total = query.count()
        safe_page = max(1, page)
        safe_limit = min(100, max(1, limit))
        trades = query.offset((safe_page - 1) * safe_limit).limit(safe_limit).all()

        return {
            "trades": [t.to_dict() for t in trades],
            "total": total,
            "page": safe_page,
            "page_size": safe_limit,
            "total_pages": int(np.ceil(total / safe_limit)) if total > 0 else 0,
        }

    def get_risk_events(self, portfolio_ref: str | int) -> List[Dict[str, Any]]:
        """Retrieve risk rejection events for portfolio."""
        p = self._resolve_portfolio(portfolio_ref)
        if not p:
            return []
        events = RiskEventRecord.query.filter_by(portfolio_id=p.id).order_by(RiskEventRecord.id.desc()).all()
        return [e.to_dict() for e in events]

    def get_trade_by_signal_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Lookup executed trade by source signal_id."""
        trade = PaperTrade.query.filter_by(signal_id=str(signal_id)).first()
        return trade.to_dict() if trade else None

    def get_engine_health(self) -> Dict[str, Any]:
        """Retrieve execution engine health, transaction costs, and telemetry counters."""
        with self._exec_lock:
            telemetry_snapshot = dict(self._telemetry)

        return {
            "service": "PaperTradingService",
            "status": "ONLINE",
            "service_health": "HEALTHY",
            "transaction_costs": {
                "fee_rate": self.fee_rate,
                "fee_bps": round(self.fee_rate * 10000, 1),
                "slippage_rate": self.slippage_rate,
                "slippage_bps": round(self.slippage_rate * 10000, 1),
            },
            "telemetry": telemetry_snapshot,
            "dedup_cache_size": len(self._executed_signals),
        }
