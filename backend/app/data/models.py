"""Validation reporting models and diagnostics for TradeMind Market Data."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimeSeriesGap:
    """Represents an observed temporal gap in the time series."""
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    expected_interval: str
    estimated_missing_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "expected_interval": self.expected_interval,
            "estimated_missing_bars": self.estimated_missing_bars,
        }


@dataclass
class RejectedRow:
    """Represents a row rejected during validation with reason."""
    index: int
    timestamp: Optional[str]
    reasons: List[str]
    raw_values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "raw_values": self.raw_values,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for an ingested market dataset.
    
    Distinguishes rejected individual rows from missing time intervals (gaps).
    Maintains zero-tolerance for silent interpolation or invented observations.
    """
    symbol: str
    source: str
    interval: Optional[str] = None
    raw_row_count: int = 0
    valid_row_count: int = 0
    rejected_row_count: int = 0
    duplicate_timestamp_count: int = 0
    missing_value_count: int = 0
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    
    # Detailed diagnostics
    rejected_rows: List[RejectedRow] = field(default_factory=list)
    detected_gaps: List[TimeSeriesGap] = field(default_factory=list)
    duplicate_timestamps: List[str] = field(default_factory=list)

    @property
    def has_rejected_rows(self) -> bool:
        return self.rejected_row_count > 0

    @property
    def has_gaps(self) -> bool:
        return len(self.detected_gaps) > 0

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_timestamp_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "symbol": self.symbol,
            "source": self.source,
            "interval": self.interval,
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "rejected_row_count": self.rejected_row_count,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "missing_value_count": self.missing_value_count,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "has_rejected_rows": self.has_rejected_rows,
            "has_gaps": self.has_gaps,
            "has_duplicates": self.has_duplicates,
            "detected_gaps": [gap.to_dict() for gap in self.detected_gaps],
            "rejected_rows": [r.to_dict() for r in self.rejected_rows],
            "duplicate_timestamps": self.duplicate_timestamps,
        }

    def summary(self) -> str:
        """Produce a human-readable diagnostic summary string."""
        lines = [
            f"=== TradeMind Data Validation Report ===",
            f"Symbol:               {self.symbol}",
            f"Source:               {self.source}",
            f"Interval:             {self.interval or 'Unspecified'}",
            f"Raw Rows:             {self.raw_row_count}",
            f"Valid Rows:           {self.valid_row_count}",
            f"Rejected Rows:        {self.rejected_row_count}",
            f"Duplicate Timestamps: {self.duplicate_timestamp_count}",
            f"Missing Values:       {self.missing_value_count}",
            f"Time Range:           {self.start_timestamp or 'N/A'} -> {self.end_timestamp or 'N/A'}",
            f"Gaps Detected:        {len(self.detected_gaps)}",
        ]
        
        if self.rejected_rows:
            lines.append("--- Sample Rejected Rows (up to 5) ---")
            for r in self.rejected_rows[:5]:
                lines.append(f"  Row {r.index} ({r.timestamp or 'No Timestamp'}): {', '.join(r.reasons)}")
                
        if self.detected_gaps:
            lines.append("--- Sample Time Series Gaps (up to 5) ---")
            for g in self.detected_gaps[:5]:
                lines.append(f"  Gap from {g.start_timestamp} to {g.end_timestamp} "
                             f"(~{g.estimated_missing_bars} missing bars, {g.duration_seconds/3600:.1f}h)")
                
        lines.append("========================================")
        return "\n".join(lines)
