"""TradeMind Real-Time Market Data Integration Package (Phase 11A - 11G).

Exports models, providers, ingestion service, candle builder, protocol parser,
feature engineering pipeline, and centralized market data manager.
"""
from .base_provider import RealtimeDataProvider
from .candle_builder import Candle, CandleBuilderService
from .feature_pipeline import LiveFeaturePipeline
from .ingestion import LiveMarketDataIngestionService
from .kite_provider import KiteRealtimeProvider
from .manager import RealtimeMarketDataManager
from .mock_provider import MockRealtimeProvider
from .models import (
    ConnectionStatus,
    ConnectionTelemetry,
    MarketQuote,
    ProviderConfig,
)
from .parser import KiteProtocolParser
from .subscription import (
    InstrumentActivityStatus,
    InstrumentSubscription,
    SubscriptionManager,
    SubscriptionState,
)
from .validator import RealtimeDataValidator

__all__ = [
    "ConnectionStatus",
    "ConnectionTelemetry",
    "MarketQuote",
    "ProviderConfig",
    "RealtimeDataValidator",
    "RealtimeDataProvider",
    "KiteRealtimeProvider",
    "MockRealtimeProvider",
    "RealtimeMarketDataManager",
    "KiteProtocolParser",
    "SubscriptionManager",
    "SubscriptionState",
    "InstrumentActivityStatus",
    "InstrumentSubscription",
    "LiveMarketDataIngestionService",
    "Candle",
    "CandleBuilderService",
    "LiveFeaturePipeline",
]
