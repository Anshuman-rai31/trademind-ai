"""End-to-End Trading Pipeline Service for TradeMind (Phase 9.5B).

Connects:
    Historical Market Data -> Feature Engineering -> Regime Detection ->
    PPO Policy Inference -> Risk Management Gate -> Portfolio Simulator Execution ->
    SQLite Persistence -> API / Frontend Telemetry.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.validator import validate_symbol
from ..db import db, safe_json_dumps, safe_serialize
from ..features.engineer import FeatureConfig, apply_all_features
from ..models.paper_trading import (
    PaperPortfolio,
    PaperPosition,
    PaperSnapshotRecord,
    PaperTrade,
    PortfolioBucketItem,
    RiskEventRecord,
)
from ..risk.config import RiskConfig
from ..risk.engine import RiskManager
from ..risk.models import InterventionStatus
from ..simulation.models import Order, OrderSide
from ..simulation.portfolio import Portfolio
from .market_data_service import MarketDataService
from .model_registry_service import ModelRegistryService
from .regime_service import RegimeService

logger = logging.getLogger(__name__)

SUPPORTED_NSE_SYMBOLS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
]


class TradingPipelineService:
    """Orchestrates end-to-end quantitative simulation, inference, risk evaluation, and persistence."""

    def __init__(
        self,
        data_service: Optional[MarketDataService] = None,
        registry_service: Optional[ModelRegistryService] = None,
        regime_service: Optional[RegimeService] = None,
    ):
        self.data_service = data_service or MarketDataService()
        self.registry_service = registry_service or ModelRegistryService()
        self.regime_service = regime_service or RegimeService(data_service=self.data_service)
        self.risk_manager = RiskManager(config=RiskConfig(enabled=True))

    @staticmethod
    def get_supported_symbols() -> List[str]:
        """Return list of canonical supported NSE stock symbols."""
        return list(SUPPORTED_NSE_SYMBOLS)

    def validate_bucket_allocations(
        self,
        bucket: List[Dict[str, Any]],
        allocation_method: str = "EQUAL_WEIGHT",
    ) -> List[Dict[str, Any]]:
        """Validate stock bucket configuration and normalize target weights.
        
        Rules:
            1. Bucket must contain at least 1 symbol.
            2. All symbols must be valid and supported.
            3. No duplicate symbols in the bucket.
            4. If EQUAL_WEIGHT, weights are automatically distributed equally (1.0 / N).
            5. If CUSTOM, each weight must be > 0, and sum must equal 100% (within 1.5% tolerance),
               then normalized so the sum is exactly 1.0.
        """
        if not bucket or not isinstance(bucket, list):
            raise ValueError("Stock bucket cannot be empty. Please select at least one stock.")

        supported = set(self.get_supported_symbols())
        seen_symbols = set()
        cleaned_items: List[Dict[str, Any]] = []

        for item in bucket:
            if not isinstance(item, dict) or "symbol" not in item:
                raise ValueError("Each bucket item must be an object with a 'symbol' field.")
            raw_sym = str(item["symbol"]).strip().upper()
            try:
                clean_sym = validate_symbol(raw_sym)
            except Exception:
                raise ValueError(f"Invalid symbol format '{raw_sym}'.")

            if clean_sym not in supported:
                raise ValueError(
                    f"Symbol '{clean_sym}' is not supported. Supported symbols: {sorted(list(supported))}."
                )

            if clean_sym in seen_symbols:
                raise ValueError(f"Duplicate symbol '{clean_sym}' found in stock bucket.")
            seen_symbols.add(clean_sym)

            # Weight can be provided as 'weight' (0-1 or 0-100) or 'target_weight' or 'target_pct'
            raw_weight = item.get("target_weight", item.get("weight", item.get("target_pct", 0.0)))
            try:
                weight_val = float(raw_weight)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid target weight for symbol '{clean_sym}'. Must be a number.")

            # If user provided weight in percentage (e.g. 25.0 instead of 0.25)
            if weight_val > 1.0 and weight_val <= 100.0:
                weight_val = weight_val / 100.0

            cleaned_items.append({"symbol": clean_sym, "target_weight": weight_val})

        n = len(cleaned_items)
        alloc_method = allocation_method.upper().strip() if allocation_method else "EQUAL_WEIGHT"

        if alloc_method == "EQUAL_WEIGHT":
            eq_weight = 1.0 / n
            for it in cleaned_items:
                it["target_weight"] = round(eq_weight, 6)
        else:
            # CUSTOM
            total_w = sum(it["target_weight"] for it in cleaned_items)
            for it in cleaned_items:
                if it["target_weight"] <= 0.0:
                    raise ValueError(f"Target allocation for '{it['symbol']}' must be strictly positive (> 0%).")

            # Check that total weight is approximately 1.0 (between 0.985 and 1.015)
            if not (0.985 <= total_w <= 1.015):
                raise ValueError(
                    f"Total target allocation must equal 100% (currently {round(total_w * 100, 2)}%)."
                )

            # Normalize exactly to 1.0
            for it in cleaned_items:
                it["target_weight"] = it["target_weight"] / total_w

        return cleaned_items

    def create_portfolio(
        self,
        name: str,
        initial_cash: float,
        bucket: List[Dict[str, Any]],
        allocation_method: str = "EQUAL_WEIGHT",
        strategy_type: str = "BUY_AND_HOLD",
        risk_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        user_id: str = "default_user",
    ) -> PaperPortfolio:
        """Create a new isolated user portfolio with validated stock bucket and risk settings."""
        if not name or not str(name).strip():
            raise ValueError("Portfolio name cannot be empty.")
        clean_name = str(name).strip()
        if len(clean_name) > 128:
            raise ValueError("Portfolio name must be 128 characters or fewer.")

        existing = PaperPortfolio.query.filter_by(name=clean_name).first()
        if existing:
            raise ValueError(f"A portfolio named '{clean_name}' already exists. Please choose a unique name.")

        try:
            init_cash = float(initial_cash)
        except (ValueError, TypeError):
            raise ValueError("Initial capital must be a valid number.")

        if init_cash < 1000.0:
            raise ValueError("Initial virtual capital must be at least ₹1,000.00.")

        alloc_method = allocation_method.upper().strip() if allocation_method else "EQUAL_WEIGHT"
        if alloc_method not in ("EQUAL_WEIGHT", "CUSTOM"):
            raise ValueError("Allocation method must be either 'EQUAL_WEIGHT' or 'CUSTOM'.")

        supported_strategies = {"BUY_AND_HOLD", "PPO_AGENT", "CUSTOM_WEIGHTS", "MA_CROSSOVER"}
        clean_strategy = strategy_type.upper().strip() if strategy_type else "BUY_AND_HOLD"
        if clean_strategy not in supported_strategies:
            raise ValueError(
                f"Strategy '{strategy_type}' is not supported. Supported: {sorted(list(supported_strategies))}."
            )

        validated_bucket = self.validate_bucket_allocations(bucket, allocation_method=alloc_method)

        portfolio = PaperPortfolio(
            name=clean_name,
            initial_cash=init_cash,
            cash=init_cash,
            portfolio_value=init_cash,
            peak_value=init_cash,
            total_realized_pnl=0.0,
            total_trades_count=0,
            allocation_method=alloc_method,
            strategy_type=clean_strategy,
            risk_config_json=safe_json_dumps(risk_config or {}),
            description=str(description).strip() if description else None,
            user_id=str(user_id).strip() if user_id else "default_user",
        )
        db.session.add(portfolio)
        db.session.flush()

        for b_item in validated_bucket:
            bucket_rec = PortfolioBucketItem(
                portfolio_id=portfolio.id,
                symbol=b_item["symbol"],
                target_weight=b_item["target_weight"],
            )
            db.session.add(bucket_rec)

        db.session.commit()
        return portfolio

    def get_portfolio(self, name_or_id: Union[str, int]) -> Optional[PaperPortfolio]:
        """Look up portfolio by integer ID or unique name."""
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
            return PaperPortfolio.query.filter_by(id=int(name_or_id)).first()
        return PaperPortfolio.query.filter_by(name=str(name_or_id)).first()

    def list_portfolios(self, user_id: Optional[str] = None) -> List[PaperPortfolio]:
        """List all configured portfolios, ensuring default portfolio exists."""
        self.get_or_create_portfolio(name="default")
        query = PaperPortfolio.query
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.order_by(PaperPortfolio.id.asc()).all()

    def update_portfolio_bucket(
        self,
        name_or_id: Union[str, int],
        bucket: List[Dict[str, Any]],
        allocation_method: Optional[str] = None,
    ) -> PaperPortfolio:
        """Update the stock selection bucket and target allocations of an existing portfolio."""
        portfolio = self.get_portfolio(name_or_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{name_or_id}' not found.")

        alloc_method = allocation_method or portfolio.allocation_method
        validated_bucket = self.validate_bucket_allocations(bucket, allocation_method=alloc_method)

        # Remove existing bucket items
        PortfolioBucketItem.query.filter_by(portfolio_id=portfolio.id).delete()

        # Add updated bucket items
        for b_item in validated_bucket:
            item = PortfolioBucketItem(
                portfolio_id=portfolio.id,
                symbol=b_item["symbol"],
                target_weight=b_item["target_weight"],
            )
            db.session.add(item)

        if allocation_method:
            portfolio.allocation_method = alloc_method.upper().strip()

        portfolio.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return portfolio

    def delete_portfolio(self, name_or_id: Union[str, int]) -> bool:
        """Delete a custom portfolio by ID or name (cannot delete 'default')."""
        portfolio = self.get_portfolio(name_or_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{name_or_id}' not found.")
        if portfolio.name.lower() == "default":
            raise ValueError("The 'default' portfolio cannot be deleted.")

        db.session.delete(portfolio)
        db.session.commit()
        return True

    def get_or_create_portfolio(self, name: str = "default", initial_cash: float = 100000.0) -> PaperPortfolio:
        """Retrieve existing simulation portfolio by name or initialize a new one."""
        portfolio = PaperPortfolio.query.filter_by(name=name).first()
        if not portfolio:
            portfolio = PaperPortfolio(
                name=name,
                initial_cash=initial_cash,
                cash=initial_cash,
                portfolio_value=initial_cash,
                peak_value=initial_cash,
                allocation_method="EQUAL_WEIGHT",
                strategy_type="BUY_AND_HOLD",
            )
            db.session.add(portfolio)
            db.session.flush()

            # Seed default 5-asset equal-weight bucket for default portfolio
            for sym in SUPPORTED_NSE_SYMBOLS:
                bucket_rec = PortfolioBucketItem(
                    portfolio_id=portfolio.id,
                    symbol=sym,
                    target_weight=1.0 / len(SUPPORTED_NSE_SYMBOLS),
                )
                db.session.add(bucket_rec)

            db.session.commit()
        return portfolio

    def execute_simulation_step(
        self,
        symbol: str,
        portfolio_name: str = "default",
        step_bar_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a single deterministic simulation step on the historical series.
        
        Steps:
            1. Load validated historical OHLCV data.
            2. Compute rolling technical indicators up to decision bar t (no future leakage).
            3. Query market regime detector (if fitted and available).
            4. Fetch active approved PPO model from registry.
            5. Assemble canonical 12-dimensional observation vector.
            6. Run PPO policy inference -> Action in {0: HOLD, 1: BUY, 2: SELL}.
            7. Pass proposed trade to institutional RiskManager.
            8. Execute effective action against simulation portfolio.
            9. Commit PaperTrade, PaperPosition, and RiskEventRecord in an atomic DB transaction.
            10. Return detailed structured telemetry.
        """
        clean_sym = validate_symbol(symbol)

        # 1. Load validated historical dataset
        df = self.data_service.get_historical_dataframe(clean_sym)
        if df is None or len(df) < 55:
            raise ValueError(
                f"Insufficient historical data for symbol '{clean_sym}'. "
                f"A minimum of 55 bars is required for rolling indicators and warmup."
            )

        # Target bar index (default to the latest available bar)
        if step_bar_idx is not None:
            if step_bar_idx < 50 or step_bar_idx >= len(df):
                raise ValueError(f"step_bar_idx {step_bar_idx} out of range [50, {len(df) - 1}]")
            curr_idx = step_bar_idx
        else:
            curr_idx = len(df) - 1

        # 2. Feature engineering strictly on data up to curr_idx (zero future lookahead)
        window_df = df.iloc[: curr_idx + 1].copy()
        feat_cfg = FeatureConfig(
            rsi_period=14,
            bb_window=20,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            atr_period=14,
        )
        featured_df = apply_all_features(window_df, config=feat_cfg)
        target_bar = featured_df.iloc[-1]
        decision_timestamp = str(target_bar["timestamp"])
        current_price = float(target_bar["close"])

        # 3. Market Regime Detection (graceful degradation if missing)
        regime_info = None
        try:
            regime_latest = self.regime_service.get_latest_regime(clean_sym)
            if regime_latest and regime_latest.get("model_available"):
                regime_info = {
                    "cluster_id": regime_latest.get("cluster_id"),
                    "regime_label": regime_latest.get("regime_label"),
                }
        except Exception as exc:
            logger.debug("Regime detector not loaded for %s: %s", clean_sym, exc)

        # 4. Model Registry: Fetch active approved PPO model
        # Make sure local artifacts are registered
        self.registry_service.sync_local_artifacts()
        model, model_rec = self.registry_service.load_model_for_inference(clean_sym)

        # 5. Fetch simulation portfolio
        portfolio = self.get_or_create_portfolio(name=portfolio_name)
        pos_rec = PaperPosition.query.filter_by(portfolio_id=portfolio.id, symbol=clean_sym).first()
        current_holding = pos_rec.quantity if pos_rec else 0.0
        avg_entry = pos_rec.average_entry_price if pos_rec else 0.0

        # Construct 12-dimensional observation vector
        market_feats = [
            float(target_bar.get("simple_return", 0.0) or 0.0),
            float(target_bar.get("log_return", 0.0) or 0.0),
            float(target_bar.get("rsi_14", 50.0) or 50.0),
            float(target_bar.get("bb_pct_20_2.0", 0.5) or 0.5),
            float(target_bar.get("macd_histogram", 0.0) or 0.0),
            float(target_bar.get("atr_14", 0.0) or 0.0),
            float(target_bar.get("close_zscore_20", 0.0) or 0.0),
        ]

        total_value = portfolio.cash + (current_holding * current_price)
        safe_val = total_value if total_value > 0 else portfolio.initial_cash
        cash_ratio = np.clip(portfolio.cash / safe_val, 0.0, 1.0)
        holdings_ratio = np.clip((current_holding * current_price) / safe_val, 0.0, 1.0)
        pos_flag = 1.0 if current_holding > 0 else 0.0
        unrealized_ret = (current_price - avg_entry) / avg_entry if current_holding > 0 and avg_entry > 0 else 0.0
        drawdown = max(0.0, (portfolio.peak_value - total_value) / portfolio.peak_value) if portfolio.peak_value > 0 else 0.0

        portfolio_feats = [
            float(cash_ratio),
            float(holdings_ratio),
            float(pos_flag),
            float(unrealized_ret),
            float(drawdown),
        ]

        observation = np.array(market_feats + portfolio_feats, dtype=np.float32)

        # 6. PPO Policy Inference
        raw_action, _ = model.predict(observation, deterministic=True)
        action_int = int(raw_action)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        proposed_action_name = action_map.get(action_int, "HOLD")

        # 7. Pre-Trade Risk Evaluation
        proposed_side = OrderSide.BUY if proposed_action_name == "BUY" else OrderSide.SELL
        allocation_pct = 0.95
        target_capital = portfolio.cash * allocation_pct
        proposed_qty = np.floor(target_capital / current_price) if proposed_action_name == "BUY" else current_holding

        risk_status = "APPROVED"
        effective_action_name = proposed_action_name
        effective_qty = proposed_qty
        risk_event_record = None

        if proposed_action_name in ("BUY", "SELL") and proposed_qty > 0:
            order = Order(
                symbol=clean_sym,
                side=proposed_side,
                quantity=float(proposed_qty),
                created_at=pd.Timestamp(decision_timestamp),
            )
            sim_portfolio = Portfolio(initial_cash=portfolio.initial_cash)
            sim_portfolio.cash = portfolio.cash
            sim_portfolio.peak_value = portfolio.peak_value
            if current_holding > 0:
                sim_portfolio.holdings[clean_sym] = current_holding
                sim_portfolio.cost_basis[clean_sym] = avg_entry

            approved_order, intervention = self.risk_manager.evaluate_proposed_order(
                order=order,
                portfolio=sim_portfolio,
                execution_price=current_price,
                timestamp=decision_timestamp,
            )

            if intervention is not None:
                if approved_order is None or approved_order.quantity <= 0:
                    risk_status = "BLOCKED"
                    effective_action_name = "HOLD"
                    effective_qty = 0.0
                else:
                    risk_status = "MODIFIED"
                    effective_qty = approved_order.quantity

                rule_name = (
                    intervention.reason.value
                    if hasattr(intervention.reason, "value")
                    else str(intervention.reason)
                )
                risk_event_record = RiskEventRecord(
                    portfolio_id=portfolio.id,
                    symbol=clean_sym,
                    timestamp=decision_timestamp,
                    rule_breached=rule_name,
                    severity="BLOCKED" if risk_status == "BLOCKED" else "INTERVENTION",
                    proposed_action=f"{proposed_action_name} {proposed_qty:.0f}",
                    effective_action=f"{effective_action_name} {effective_qty:.0f}",
                    details=intervention.details,
                )
                db.session.add(risk_event_record)
            else:
                risk_status = "APPROVED"
                effective_action_name = proposed_action_name
                effective_qty = proposed_qty

        # 8. Portfolio Simulator Execution & Accounting
        executed_trade = None
        fee_rate = 0.0003
        slippage_rate = 0.0005

        if effective_action_name == "BUY" and effective_qty > 0:
            fill_price = current_price * (1.0 + slippage_rate)
            gross_cost = effective_qty * fill_price
            fee = gross_cost * fee_rate
            total_outlay = gross_cost + fee

            if portfolio.cash >= total_outlay:
                portfolio.cash -= total_outlay
                new_qty = current_holding + effective_qty
                # Update weighted average cost basis
                new_avg_entry = ((current_holding * avg_entry) + gross_cost) / new_qty
                if not pos_rec:
                    pos_rec = PaperPosition(portfolio_id=portfolio.id, symbol=clean_sym)
                    db.session.add(pos_rec)
                pos_rec.quantity = new_qty
                pos_rec.average_entry_price = new_avg_entry
                pos_rec.current_price = current_price
                pos_rec.unrealized_pnl = (current_price - new_avg_entry) * new_qty

                executed_trade = PaperTrade(
                    portfolio_id=portfolio.id,
                    symbol=clean_sym,
                    timestamp=decision_timestamp,
                    action="BUY",
                    quantity=effective_qty,
                    price=fill_price,
                    fee=fee,
                    slippage=effective_qty * current_price * slippage_rate,
                    realized_pnl=0.0,
                    risk_status=risk_status,
                    metadata_json=safe_json_dumps({"model_id": model_rec.model_id}),
                )
                db.session.add(executed_trade)
                portfolio.total_trades_count += 1

        elif effective_action_name == "SELL" and current_holding > 0:
            fill_price = current_price * (1.0 - slippage_rate)
            sell_qty = min(current_holding, effective_qty)
            gross_proceeds = sell_qty * fill_price
            fee = gross_proceeds * fee_rate
            net_proceeds = gross_proceeds - fee

            realized_pnl = (fill_price - avg_entry) * sell_qty - fee
            portfolio.cash += net_proceeds
            portfolio.total_realized_pnl += realized_pnl

            remaining_qty = current_holding - sell_qty
            if pos_rec:
                pos_rec.quantity = remaining_qty
                pos_rec.current_price = current_price
                pos_rec.unrealized_pnl = (current_price - avg_entry) * remaining_qty if remaining_qty > 0 else 0.0

            executed_trade = PaperTrade(
                portfolio_id=portfolio.id,
                symbol=clean_sym,
                timestamp=decision_timestamp,
                action="SELL",
                quantity=sell_qty,
                price=fill_price,
                fee=fee,
                slippage=sell_qty * current_price * slippage_rate,
                realized_pnl=realized_pnl,
                risk_status=risk_status,
                metadata_json=safe_json_dumps({"model_id": model_rec.model_id}),
            )
            db.session.add(executed_trade)
            portfolio.total_trades_count += 1

        # Mark-to-market portfolio value
        final_holdings_val = (pos_rec.quantity if pos_rec else 0.0) * current_price
        portfolio.portfolio_value = portfolio.cash + final_holdings_val
        if portfolio.portfolio_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.portfolio_value
        portfolio.updated_at = datetime.now(timezone.utc)

        # Record valuation snapshot for equity curve
        step_dd = max(0.0, (portfolio.peak_value - portfolio.portfolio_value) / portfolio.peak_value) if portfolio.peak_value > 0 else 0.0
        snapshot_rec = PaperSnapshotRecord(
            portfolio_id=portfolio.id,
            timestamp=decision_timestamp,
            cash=portfolio.cash,
            holdings_value=final_holdings_val,
            portfolio_value=portfolio.portfolio_value,
            drawdown=step_dd,
        )
        db.session.add(snapshot_rec)

        # 9. Atomic DB Commit
        db.session.commit()

        # 10. Assemble structured output
        return {
            "success": True,
            "timestamp": decision_timestamp,
            "symbol": clean_sym,
            "current_price": round(current_price, 2),
            "model": {
                "model_id": model_rec.model_id,
                "version": model_rec.version,
                "algorithm": model_rec.algorithm,
            },
            "regime": regime_info,
            "decision": {
                "proposed_action": proposed_action_name,
                "effective_action": effective_action_name,
                "risk_status": risk_status,
                "action_id": action_int,
            },
            "trade_executed": executed_trade.to_dict() if executed_trade else None,
            "portfolio": portfolio.to_dict(include_positions=True),
            "risk_intervention": risk_event_record.to_dict() if risk_event_record else None,
            "snapshot": snapshot_rec.to_dict(),
        }

    def get_portfolio_snapshots(self, name_or_id: Union[str, int]) -> List[PaperSnapshotRecord]:
        """Retrieve valuation snapshots for the given portfolio."""
        portfolio = self.get_portfolio(name_or_id)
        if not portfolio:
            return []
        return (
            PaperSnapshotRecord.query.filter_by(portfolio_id=portfolio.id)
            .order_by(PaperSnapshotRecord.id.asc())
            .all()
        )

    def rebalance_portfolio(
        self,
        name_or_id: Union[str, int],
        step_bar_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute simulated multi-stock rebalancing according to portfolio bucket target allocations.
        
        Algorithm:
            1. Validate portfolio and ensure bucket has configured target stocks.
            2. Fetch latest market close price and timestamp for every stock in the bucket.
            3. Calculate current total mark-to-market portfolio value: Cash + sum(holdings_i * price_i).
            4. Compute target capital per stock: T_i = Portfolio_Value * target_weight_i.
            5. Target shares per stock: Target_Shares_i = floor(T_i / price_i).
            6. Delta shares: Delta_i = Target_Shares_i - Current_Shares_i.
            7. Stage 1 (SELL Orders): Execute sells first to release cash.
               - Fill with 0.05% slippage and 0.03% fee.
               - Credit proceeds to cash, record realized PnL and PaperTrade.
            8. Stage 2 (BUY Orders): Execute buys using available cash.
               - Check available cash and adjust quantities if necessary.
               - Fill with 0.05% slippage and 0.03% fee.
               - Debit cash, update weighted average entry price and PaperTrade.
            9. Update portfolio total value, peak value, and record PaperSnapshotRecord.
            10. Atomic DB commit and return execution summary.
        """
        portfolio = self.get_portfolio(name_or_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{name_or_id}' not found.")

        bucket_items = portfolio.bucket_items
        if not bucket_items:
            raise ValueError(f"Portfolio '{portfolio.name}' has no stocks in its selection bucket.")

        # 1. Fetch current price and latest timestamp for each symbol
        stock_data: Dict[str, Dict[str, Any]] = {}
        decision_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for item in bucket_items:
            sym = item.symbol
            df = self.data_service.get_historical_dataframe(sym)
            if df is None or len(df) == 0:
                raise ValueError(f"Historical market data unavailable for bucket symbol '{sym}'.")

            bar_pos = step_bar_idx if (step_bar_idx is not None and 0 <= step_bar_idx < len(df)) else -1
            bar = df.iloc[bar_pos]
            price = float(bar["close"])
            if price <= 0:
                raise ValueError(f"Invalid market close price for '{sym}': {price}")

            stock_data[sym] = {
                "price": price,
                "timestamp": str(bar["timestamp"]),
                "weight": float(item.target_weight),
            }
            decision_timestamp = str(bar["timestamp"])

        # 2. Compute current mark-to-market portfolio value
        positions_map: Dict[str, PaperPosition] = {p.symbol: p for p in portfolio.positions}
        current_holdings_val = 0.0
        for sym, pos in positions_map.items():
            sym_price = stock_data.get(sym, {}).get("price")
            if sym_price is None:
                df_extra = self.data_service.get_historical_dataframe(sym)
                sym_price = (
                    float(df_extra.iloc[-1]["close"])
                    if df_extra is not None and len(df_extra) > 0
                    else pos.current_price
                )
            current_holdings_val += pos.quantity * sym_price

        total_portfolio_value = portfolio.cash + current_holdings_val
        if total_portfolio_value <= 0:
            total_portfolio_value = portfolio.initial_cash

        # 3. Determine target shares and deltas for bucket stocks
        orders_to_place: List[Dict[str, Any]] = []
        for item in bucket_items:
            sym = item.symbol
            s_info = stock_data[sym]
            p = s_info["price"]
            w = s_info["weight"]
            target_cap = total_portfolio_value * w
            target_shares = float(np.floor(target_cap / p))
            curr_pos = positions_map.get(sym)
            curr_shares = curr_pos.quantity if curr_pos else 0.0
            delta = target_shares - curr_shares

            if abs(delta) >= 1.0:
                orders_to_place.append({
                    "symbol": sym,
                    "side": "BUY" if delta > 0 else "SELL",
                    "delta_qty": abs(delta),
                    "price": p,
                    "target_shares": target_shares,
                    "curr_shares": curr_shares,
                })

        fee_rate = 0.0003
        slippage_rate = 0.0005
        executed_trades: List[PaperTrade] = []

        # 4. Stage 1: Execute SELL orders first to free up capital
        sells = [o for o in orders_to_place if o["side"] == "SELL"]
        for o in sells:
            sym = o["symbol"]
            curr_pos = positions_map.get(sym)
            if not curr_pos or curr_pos.quantity <= 0:
                continue
            sell_qty = min(curr_pos.quantity, o["delta_qty"])
            if sell_qty <= 0:
                continue

            current_price = o["price"]
            fill_price = current_price * (1.0 - slippage_rate)
            gross = sell_qty * fill_price
            fee = gross * fee_rate
            net = gross - fee

            avg_entry = curr_pos.average_entry_price
            realized_pnl = (fill_price - avg_entry) * sell_qty - fee

            portfolio.cash += net
            portfolio.total_realized_pnl += realized_pnl

            curr_pos.quantity -= sell_qty
            curr_pos.current_price = current_price
            curr_pos.unrealized_pnl = (
                (current_price - avg_entry) * curr_pos.quantity if curr_pos.quantity > 0 else 0.0
            )

            trade = PaperTrade(
                portfolio_id=portfolio.id,
                symbol=sym,
                timestamp=stock_data[sym]["timestamp"],
                action="SELL",
                quantity=sell_qty,
                price=fill_price,
                fee=fee,
                slippage=sell_qty * current_price * slippage_rate,
                realized_pnl=realized_pnl,
                risk_status="APPROVED",
                metadata_json=safe_json_dumps({"reason": "REBALANCE"}),
            )
            db.session.add(trade)
            portfolio.total_trades_count += 1
            executed_trades.append(trade)

        # 5. Stage 2: Execute BUY orders with available cash
        buys = [o for o in orders_to_place if o["side"] == "BUY"]
        for o in buys:
            sym = o["symbol"]
            current_price = o["price"]
            fill_price = current_price * (1.0 + slippage_rate)
            cost_per_share = fill_price * (1.0 + fee_rate)

            desired_qty = o["delta_qty"]
            max_affordable_qty = np.floor(portfolio.cash / cost_per_share) if cost_per_share > 0 else 0.0
            buy_qty = min(desired_qty, max_affordable_qty)

            if buy_qty <= 0:
                continue

            gross = buy_qty * fill_price
            fee = gross * fee_rate
            total_outlay = gross + fee

            if portfolio.cash < total_outlay:
                continue

            portfolio.cash -= total_outlay
            curr_pos = positions_map.get(sym)
            if not curr_pos:
                curr_pos = PaperPosition(portfolio_id=portfolio.id, symbol=sym)
                db.session.add(curr_pos)
                positions_map[sym] = curr_pos
                old_qty = 0.0
                old_avg = 0.0
            else:
                old_qty = curr_pos.quantity
                old_avg = curr_pos.average_entry_price

            new_qty = old_qty + buy_qty
            new_avg = ((old_qty * old_avg) + gross) / new_qty if new_qty > 0 else fill_price

            curr_pos.quantity = new_qty
            curr_pos.average_entry_price = new_avg
            curr_pos.current_price = current_price
            curr_pos.unrealized_pnl = (current_price - new_avg) * new_qty

            trade = PaperTrade(
                portfolio_id=portfolio.id,
                symbol=sym,
                timestamp=stock_data[sym]["timestamp"],
                action="BUY",
                quantity=buy_qty,
                price=fill_price,
                fee=fee,
                slippage=buy_qty * current_price * slippage_rate,
                realized_pnl=0.0,
                risk_status="APPROVED",
                metadata_json=safe_json_dumps({"reason": "REBALANCE"}),
            )
            db.session.add(trade)
            portfolio.total_trades_count += 1
            executed_trades.append(trade)

        # 6. Re-mark portfolio value & record snapshot
        final_holdings = sum(
            (p.quantity * stock_data.get(p.symbol, {}).get("price", p.current_price))
            for p in portfolio.positions
            if p.quantity > 0
        )
        portfolio.portfolio_value = portfolio.cash + final_holdings
        if portfolio.portfolio_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.portfolio_value
        portfolio.updated_at = datetime.now(timezone.utc)

        dd = (
            max(0.0, (portfolio.peak_value - portfolio.portfolio_value) / portfolio.peak_value)
            if portfolio.peak_value > 0
            else 0.0
        )
        snapshot = PaperSnapshotRecord(
            portfolio_id=portfolio.id,
            timestamp=decision_timestamp,
            cash=portfolio.cash,
            holdings_value=final_holdings,
            portfolio_value=portfolio.portfolio_value,
            drawdown=dd,
        )
        db.session.add(snapshot)

        db.session.commit()

        return {
            "success": True,
            "message": f"Portfolio '{portfolio.name}' rebalanced successfully across {len(bucket_items)} bucket assets.",
            "timestamp": decision_timestamp,
            "trades_count": len(executed_trades),
            "trades": [t.to_dict() for t in executed_trades],
            "portfolio": portfolio.to_dict(include_positions=True, include_bucket=True),
            "snapshot": snapshot.to_dict(),
        }
