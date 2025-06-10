"""
Checkpoint management system for Kura.

This module provides different checkpoint backends for storing and loading
intermediate pipeline results. The available backends are:

- JSONLCheckpointManager: Traditional JSONL file-based checkpoints (default)
- ParquetCheckpointManager: Parquet-based checkpoints for better compression
- HFDatasetCheckpointManager: HuggingFace datasets-based checkpoints

The ParquetCheckpointManager provides better compression and faster loading
for analytical workloads, while HFDatasetCheckpointManager provides
advanced features like streaming, versioning, and cloud storage integration.
"""

from .jsonl import JSONLCheckpointManager
from .hf_dataset import HFDatasetCheckpointManager
from .migration import migrate_jsonl_to_hf_dataset

# Import ParquetCheckpointManager if PyArrow is available
try:
    from .parquet import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

__all__ = [
    "JSONLCheckpointManager",
    "HFDatasetCheckpointManager",
    "migrate_jsonl_to_hf_dataset",
]

# Add ParquetCheckpointManager to exports if available
if PARQUET_AVAILABLE:
    __all__.append("ParquetCheckpointManager")
