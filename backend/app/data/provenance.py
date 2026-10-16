"""TradeMind Data Provenance and Metadata Tracker.

Records audit trails, download parameters, corporate action adjustments,
and validation results for every market data asset.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceMetadata:
    """Full provenance and lineage record for a downloaded/processed market dataset."""
    symbol: str
    source: str
    frequency: str
    auto_adjust: bool
    download_status: str  # SUCCESS, CACHED, REFRESHED, FAILED
    validation_status: str  # PASSED, WARNINGS, FAILED
    requested_start_date: str
    requested_end_date: str
    actual_min_date: Optional[str] = None
    actual_max_date: Optional[str] = None
    row_count: int = 0
    download_timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    raw_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None
    adjustment_policy_notes: str = ""
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata record to a serializable dictionary."""
        return asdict(self)

    def save(self, metadata_dir: str | Path) -> Path:
        """Write provenance record as formatted JSON file in metadata directory."""
        dir_path = Path(metadata_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{self.symbol.lower().replace('.', '_')}_{self.frequency.lower()}_provenance.json"
        dest_path = dir_path / filename

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return dest_path

    @classmethod
    def load(cls, file_path: str | Path) -> "ProvenanceMetadata":
        """Load provenance record from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
