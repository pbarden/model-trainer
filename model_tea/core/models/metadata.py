"""
Model metadata and version structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


# Simplified: Use simple strings instead of enum
# Valid statuses: "training", "validation", "staging", "production", "retired", "failed"


@dataclass
class ModelMetadata:
    """Simplified model metadata container."""
    model_id: str
    version: str
    status: str = "training"
    metrics: Dict[str, float] = None
    created_at: datetime = None
    # Store everything else in a simple 'extra' dict for flexibility
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.extra is None:
            self.extra = {}


@dataclass
class ModelVersion:
    """Simplified model version."""
    metadata: ModelMetadata
    model_path: str
    checksum: str
    # Optional paths stored in metadata.extra if needed