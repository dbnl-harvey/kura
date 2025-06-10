"""
Checkpoint management system for Kura.

This module provides different checkpoint backends for storing and loading
intermediate pipeline results. The available backends are:

- JSONLCheckpointManager: Traditional JSONL file-based checkpoints (default)
- HFDatasetCheckpointManager: HuggingFace datasets-based checkpoints

The HFDatasetCheckpointManager provides better performance, scalability,
and features like streaming, versioning, and cloud storage integration.
"""

from .jsonl import JSONLCheckpointManager
from .hf_dataset import HFDatasetCheckpointManager
from .migration import migrate_jsonl_to_hf_dataset

__all__ = [
    "JSONLCheckpointManager",
    "HFDatasetCheckpointManager", 
    "migrate_jsonl_to_hf_dataset",
]