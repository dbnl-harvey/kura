"""
Procedural implementation of the Kura conversation analysis pipeline.

This module provides a functional approach to conversation analysis, breaking down
the pipeline into composable functions that can be used independently or together.

Key benefits over the class-based approach:
- Better composability and flexibility
- Easier testing of individual steps
- Clearer data flow and dependencies
- Better support for functional programming patterns
- Support for heterogeneous models through polymorphism
"""

import logging
from typing import Optional, TypeVar, List, Union, Literal
import os
from pydantic import BaseModel

# Import existing Kura components
from kura.base_classes import (
    BaseSummaryModel,
    BaseClusterModel,
    BaseMetaClusterModel,
    BaseDimensionalityReduction,
)
from kura.types import Conversation, Cluster, ConversationSummary
from kura.types.dimensionality import ProjectedCluster

# Import checkpoint managers
from kura.checkpoints import JSONLCheckpointManager, HFDatasetCheckpointManager

# Set up logger
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

CheckpointFormat = Literal["jsonl", "hf-dataset"]


# =============================================================================
# Checkpoint Management
# =============================================================================


class CheckpointManager:
    """Unified checkpoint manager that supports multiple backend formats.
    
    This manager can use either JSONL files (legacy) or HuggingFace datasets (new)
    for storing checkpoint data. It provides a consistent interface while allowing
    users to choose the backend format based on their needs.
    """

    def __init__(
        self, 
        checkpoint_dir: str, 
        *, 
        enabled: bool = True,
        format: CheckpointFormat = "jsonl",
        # HuggingFace datasets specific options
        hub_repo: Optional[str] = None,
        hub_token: Optional[str] = None,
        streaming: bool = False,
        compression: Optional[str] = "gzip"
    ):
        """Initialize checkpoint manager with specified backend format.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
            format: Checkpoint format to use ("jsonl" or "hf-dataset")
            hub_repo: Optional HuggingFace Hub repository (HF datasets only)
            hub_token: Optional HuggingFace Hub token (HF datasets only)
            streaming: Whether to use streaming mode by default (HF datasets only)
            compression: Compression algorithm (HF datasets only)
        """
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled
        self.format = format
        
        # Initialize the appropriate backend
        if format == "jsonl":
            self._backend = JSONLCheckpointManager(checkpoint_dir, enabled=enabled)
        elif format == "hf-dataset":
            self._backend = HFDatasetCheckpointManager(
                checkpoint_dir,
                enabled=enabled,
                hub_repo=hub_repo,
                hub_token=hub_token,
                streaming=streaming,
                compression=compression
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        
        logger.info(f"Initialized {format} checkpoint manager at {checkpoint_dir}")

    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        self._backend.setup_checkpoint_dir()

    def get_checkpoint_path(self, filename: str) -> str:
        """Get full path for a checkpoint file."""
        if hasattr(self._backend, 'get_checkpoint_path'):
            return self._backend.get_checkpoint_path(filename)
        return os.path.join(self.checkpoint_dir, filename)

    def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
        """Load data from a checkpoint file if it exists.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if self.format == "hf-dataset":
            # For HF datasets, we need to determine checkpoint type
            checkpoint_type = ""
            if model_class == Conversation:
                checkpoint_type = "conversations"
            elif model_class == ConversationSummary:
                checkpoint_type = "summaries"
            elif model_class == ProjectedCluster:
                checkpoint_type = "projected_clusters"
            elif model_class == Cluster:
                checkpoint_type = "clusters"
                
            return self._backend.load_checkpoint(
                filename, 
                model_class, 
                checkpoint_type=checkpoint_type
            )
        else:
            return self._backend.load_checkpoint(filename, model_class)

    def save_checkpoint(self, filename: str, data: List[T]) -> None:
        """Save data to a checkpoint file.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
        """
        if self.format == "hf-dataset" and data:
            # For HF datasets, we need to determine checkpoint type
            model_class = type(data[0])
            checkpoint_type = ""
            if model_class == Conversation:
                checkpoint_type = "conversations"
            elif model_class == ConversationSummary:
                checkpoint_type = "summaries"
            elif model_class == ProjectedCluster:
                checkpoint_type = "projected_clusters"
            elif model_class == Cluster:
                checkpoint_type = "clusters"
                
            return self._backend.save_checkpoint(filename, data, checkpoint_type)
        else:
            return self._backend.save_checkpoint(filename, data)

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return self._backend.list_checkpoints()

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            filename: Name of the checkpoint to delete
            
        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        return self._backend.delete_checkpoint(filename)

    # Additional methods for HF datasets backend
    def get_checkpoint_info(self, filename: str) -> Optional[dict]:
        """Get information about a checkpoint (HF datasets only)."""
        if self.format == "hf-dataset" and hasattr(self._backend, 'get_checkpoint_info'):
            return self._backend.get_checkpoint_info(filename)
        return None

    def filter_checkpoint(self, filename: str, filter_fn: callable, model_class: type[T]) -> Optional[List[T]]:
        """Filter a checkpoint without loading everything into memory (HF datasets only)."""
        if self.format == "hf-dataset" and hasattr(self._backend, 'filter_checkpoint'):
            checkpoint_type = ""
            if model_class == Conversation:
                checkpoint_type = "conversations"
            elif model_class == ConversationSummary:
                checkpoint_type = "summaries"
            elif model_class == ProjectedCluster:
                checkpoint_type = "projected_clusters"
            elif model_class == Cluster:
                checkpoint_type = "clusters"
                
            return self._backend.filter_checkpoint(filename, filter_fn, model_class, checkpoint_type)
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_checkpoint_manager(
    checkpoint_dir: str = "./checkpoints",
    format: Optional[CheckpointFormat] = None,
    **kwargs
) -> CheckpointManager:
    """Create a checkpoint manager with appropriate format.
    
    This function automatically chooses the checkpoint format based on:
    1. Explicit format parameter
    2. KURA_CHECKPOINT_FORMAT environment variable
    3. Default to 'jsonl' for backward compatibility
    
    Args:
        checkpoint_dir: Directory for checkpoints
        format: Explicit format choice (overrides environment)
        **kwargs: Additional arguments passed to checkpoint manager
        
    Returns:
        Configured CheckpointManager instance
    """
    # Determine format
    if format is None:
        format = os.environ.get("KURA_CHECKPOINT_FORMAT", "jsonl")
    
    if format not in ["jsonl", "hf-dataset"]:
        logger.warning(f"Unknown checkpoint format '{format}', defaulting to 'jsonl'")
        format = "jsonl"
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        format=format,
        **kwargs
    )


def create_hf_checkpoint_manager(
    checkpoint_dir: str = "./checkpoints",
    *,
    hub_repo: Optional[str] = None,
    hub_token: Optional[str] = None,
    streaming: bool = False,
    compression: Optional[str] = "gzip",
    **kwargs
) -> CheckpointManager:
    """Create a HuggingFace datasets checkpoint manager.
    
    Convenience function for creating an HF datasets checkpoint manager
    with commonly used settings.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        hub_repo: HuggingFace Hub repository name
        hub_token: HuggingFace Hub authentication token
        streaming: Whether to use streaming mode by default
        compression: Compression algorithm to use
        **kwargs: Additional arguments
        
    Returns:
        CheckpointManager configured for HF datasets
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        format="hf-dataset",
        hub_repo=hub_repo,
        hub_token=hub_token,
        streaming=streaming,
        compression=compression,
        **kwargs
    )


# =============================================================================
# Core Pipeline Functions
# =============================================================================


async def summarise_conversations(
    conversations: List[Conversation],
    *,
    model: BaseSummaryModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[ConversationSummary]:
    """Generate summaries for a list of conversations.

    This is a pure function that takes conversations and a summary model,
    and returns conversation summaries. Optionally uses checkpointing.

    The function works with any model that implements BaseSummaryModel,
    supporting heterogeneous backends (OpenAI, vLLM, Hugging Face, etc.)
    through polymorphism.

    Args:
        conversations: List of conversations to summarize
        model: Model to use for summarization (OpenAI, vLLM, local, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of conversation summaries

    Example:
        >>> openai_model = OpenAISummaryModel(api_key="sk-...")
        >>> checkpoint_mgr = CheckpointManager("./checkpoints")
        >>> summaries = await summarise_conversations(
        ...     conversations=my_conversations,
        ...     model=openai_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting summarization of {len(conversations)} conversations using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            model.checkpoint_filename, ConversationSummary
        )
        if cached:
            logger.info(f"Loaded {len(cached)} summaries from checkpoint")
            return cached

    # Generate summaries
    logger.info("Generating new summaries...")
    summaries = await model.summarise(conversations)
    logger.info(f"Generated {len(summaries)} summaries")

    # Save to checkpoint
    if checkpoint_manager:
        logger.info(f"Saving summaries to checkpoint: {model.checkpoint_filename}")
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, summaries)

    return summaries


async def generate_base_clusters_from_conversation_summaries(
    summaries: List[ConversationSummary],
    *,
    model: BaseClusterModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]:
    """Generate base clusters from conversation summaries.

    This function groups similar summaries into initial clusters using
    the provided clustering model. Supports different clustering algorithms
    through the model interface.

    Args:
        summaries: List of conversation summaries to cluster
        model: Model to use for clustering (HDBSCAN, KMeans, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of base clusters

    Example:
        >>> cluster_model = ClusterModel(algorithm="hdbscan")
        >>> clusters = await generate_base_clusters(
        ...     summaries=conversation_summaries,
        ...     model=cluster_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting clustering of {len(summaries)} summaries using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(model.checkpoint_filename, Cluster)
        if cached:
            logger.info(f"Loaded {len(cached)} clusters from checkpoint")
            return cached

    # Generate clusters
    logger.info("Generating new clusters...")
    clusters = await model.cluster_summaries(summaries)
    logger.info(f"Generated {len(clusters)} clusters")

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, clusters)

    return clusters


async def reduce_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    model: BaseMetaClusterModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]:
    """Reduce clusters into a hierarchical structure.

    Iteratively combines similar clusters until the number of root clusters
    is less than or equal to the model's max_clusters setting.

    Args:
        clusters: List of initial clusters to reduce
        model: Meta-clustering model to use for reduction
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of clusters with hierarchical structure

    Example:
        >>> meta_model = MetaClusterModel(max_clusters=5)
        >>> reduced = await reduce_clusters(
        ...     clusters=base_clusters,
        ...     model=meta_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting cluster reduction from {len(clusters)} initial clusters using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(model.checkpoint_filename, Cluster)
        if cached:
            root_count = len([c for c in cached if c.parent_id is None])
            logger.info(
                f"Loaded {len(cached)} clusters from checkpoint ({root_count} root clusters)"
            )
            return cached

    # Start with all clusters as potential roots
    all_clusters = clusters.copy()
    root_clusters = clusters.copy()

    # Get max_clusters from model if available, otherwise use default
    max_clusters = getattr(model, "max_clusters", 10)
    logger.info(f"Starting with {len(root_clusters)} clusters, target: {max_clusters}")

    # Iteratively reduce until we have desired number of root clusters
    while len(root_clusters) > max_clusters:
        # Get updated clusters from meta-clustering
        new_current_level = await model.reduce_clusters(root_clusters)

        # Find new root clusters (those without parents)
        root_clusters = [c for c in new_current_level if c.parent_id is None]

        # Remove old clusters that now have parents
        old_cluster_ids = {c.id for c in new_current_level if c.parent_id}
        all_clusters = [c for c in all_clusters if c.id not in old_cluster_ids]

        # Add new clusters to the complete list
        all_clusters.extend(new_current_level)

        logger.info(f"Reduced to {len(root_clusters)} root clusters")

    logger.info(
        f"Cluster reduction complete: {len(all_clusters)} total clusters, {len(root_clusters)} root clusters"
    )

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, all_clusters)

    return all_clusters


async def reduce_dimensionality_from_clusters(
    clusters: List[Cluster],
    *,
    model: BaseDimensionalityReduction,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[ProjectedCluster]:
    """Reduce dimensions of clusters for visualization.

    Projects clusters to 2D space using the provided dimensionality reduction model.
    Supports different algorithms (UMAP, t-SNE, PCA, etc.) through the model interface.

    Args:
        clusters: List of clusters to project
        model: Dimensionality reduction model to use (UMAP, t-SNE, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of projected clusters with 2D coordinates

    Example:
        >>> dim_model = HDBUMAP(n_components=2)
        >>> projected = await reduce_dimensionality(
        ...     clusters=hierarchical_clusters,
        ...     model=dim_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting dimensionality reduction for {len(clusters)} clusters using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            model.checkpoint_filename, ProjectedCluster
        )
        if cached:
            logger.info(f"Loaded {len(cached)} projected clusters from checkpoint")
            return cached

    # Reduce dimensionality
    logger.info("Projecting clusters to 2D space...")
    projected_clusters = await model.reduce_dimensionality(clusters)
    logger.info(f"Projected {len(projected_clusters)} clusters to 2D")

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(
            model.checkpoint_filename, projected_clusters
        )

    return projected_clusters
