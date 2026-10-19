"""TradeMind Phase 8 Market-Regime Detection Exceptions.

Defines domain-specific exceptions for regime detection modeling, validation,
pipeline transformations, and data integrity safeguards.
"""
from __future__ import annotations


class RegimeDetectionError(Exception):
    """Base exception for all regime detection errors."""
    pass


class ModelNotFittedError(RegimeDetectionError):
    """Raised when transform, predict, or export is invoked on an unfitted detector."""
    pass


class InsufficientDataError(RegimeDetectionError):
    """Raised when training data has fewer observations than required for clustering."""
    pass


class FeatureExtractionError(RegimeDetectionError):
    """Raised when required features are missing or cannot be computed safely."""
    pass


class DataLeakageError(RegimeDetectionError):
    """Raised when a data leakage constraint is violated."""
    pass
