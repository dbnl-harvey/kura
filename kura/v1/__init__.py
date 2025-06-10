"""
Kura V1: Procedural Implementation

A functional approach to conversation analysis that breaks down the pipeline
into composable functions for better flexibility and testability.
"""

from .kura import (
    # Core pipeline functions
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    # Checkpoint management
    CheckpointManager,
    create_checkpoint_manager,
    create_hf_checkpoint_manager,
    CheckpointFormat,
)

# Import ParquetCheckpointManager if pyarrow is available
try:
    from .parquet_checkpoint import ParquetCheckpointManager
    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

__all__ = [
    # Core functions
    "summarise_conversations",
    "generate_base_clusters_from_conversation_summaries",
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
    # Utilities
    "CheckpointManager",
    "create_checkpoint_manager", 
    "create_hf_checkpoint_manager",
    "CheckpointFormat",
]

# Add ParquetCheckpointManager to __all__ if available
if PARQUET_AVAILABLE:
    __all__.append("ParquetCheckpointManager")

__version__ = "1.0.0"
