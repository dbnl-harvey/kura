"""Procedural cluster visualization utilities for Kura v1.

This module provides various methods for visualizing hierarchical cluster structures
in the terminal, including basic tree views, enhanced visualizations with statistics,
and rich-formatted output using the Rich library when available.

Compatible with the procedural Kura v1 pipeline approach.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

from kura.types import Cluster, ClusterTreeNode

# Rich package has been removed - using basic visualization only
RICH_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)


def _build_tree_structure(
    node: ClusterTreeNode,
    node_id_to_cluster: dict[str, ClusterTreeNode],
    level: int = 0,
    is_last: bool = True,
    prefix: str = "",
) -> str:
    """Build a text representation of the hierarchical cluster tree.

    This is a recursive helper function used by visualise_clusters().

    Args:
        node: Current tree node
        node_id_to_cluster: Dictionary mapping node IDs to nodes
        level: Current depth in the tree (for indentation)
        is_last: Whether this is the last child of its parent
        prefix: Current line prefix for tree structure

    Returns:
        String representation of the tree structure
    """
    # Current line prefix (used for tree visualization symbols)
    current_prefix = prefix

    # Add the appropriate connector based on whether this is the last child
    if level > 0:
        if is_last:
            current_prefix += "╚══ "
        else:
            current_prefix += "╠══ "

    # Print the current node
    result = current_prefix + node.name + " (" + str(node.count) + " conversations)\n"

    # Calculate the prefix for children (continue vertical lines for non-last children)
    child_prefix = prefix
    if level > 0:
        if is_last:
            child_prefix += "    "  # No vertical line needed for last child's children
        else:
            child_prefix += "║   "  # Continue vertical line for non-last child's children

    # Process children
    children = node.children
    for i, child_id in enumerate(children):
        child = node_id_to_cluster[child_id]
        is_last_child = i == len(children) - 1
        result += _build_tree_structure(child, node_id_to_cluster, level + 1, is_last_child, child_prefix)

    return result


def _build_enhanced_tree_structure(
    node: ClusterTreeNode,
    node_id_to_cluster: dict[str, ClusterTreeNode],
    level: int = 0,
    is_last: bool = True,
    prefix: str = "",
    total_conversations: int = 0,
) -> str:
    """Build an enhanced text representation with colors and better formatting.

    Args:
        node: Current tree node
        node_id_to_cluster: Dictionary mapping node IDs to nodes
        level: Current depth in the tree (for indentation)
        is_last: Whether this is the last child of its parent
        prefix: Current line prefix for tree structure
        total_conversations: Total conversations for percentage calculation

    Returns:
        String representation of the enhanced tree structure
    """
    # Color scheme based on level
    colors = [
        "bright_cyan",
        "bright_green",
        "bright_yellow",
        "bright_magenta",
        "bright_blue",
    ]
    colors[level % len(colors)]

    # Current line prefix (used for tree visualization symbols)
    current_prefix = prefix

    # Add the appropriate connector based on whether this is the last child
    if level > 0:
        if is_last:
            current_prefix += "╚══ "
        else:
            current_prefix += "╠══ "

    # Calculate percentage of total conversations
    percentage = (node.count / total_conversations * 100) if total_conversations > 0 else 0

    # Create progress bar for visual representation
    bar_width = 20
    filled_width = int((node.count / total_conversations) * bar_width) if total_conversations > 0 else 0
    progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)

    # Build the line with enhanced formatting
    result = f"{current_prefix}🔸 {node.name}\n"
    result += f"{prefix}{'║   ' if not is_last and level > 0 else '    '}📊 {node.count:,} conversations ({percentage:.1f}%) [{progress_bar}]\n"

    # Add description if available and not too long
    if hasattr(node, "description") and node.description and len(node.description) < 100:
        result += f"{prefix}{'║   ' if not is_last and level > 0 else '    '}💭 {node.description}\n"

    result += "\n"

    # Calculate the prefix for children
    child_prefix = prefix
    if level > 0:
        if is_last:
            child_prefix += "    "
        else:
            child_prefix += "║   "

    # Process children
    children = node.children
    for i, child_id in enumerate(children):
        child = node_id_to_cluster[child_id]
        is_last_child = i == len(children) - 1
        result += _build_enhanced_tree_structure(
            child,
            node_id_to_cluster,
            level + 1,
            is_last_child,
            child_prefix,
            total_conversations,
        )

    return result


def _load_clusters_from_checkpoint(checkpoint_path: Union[str, Path]) -> List[Cluster]:
    """Load clusters from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        List of clusters loaded from the checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint file is malformed
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        with open(checkpoint_path) as f:
            clusters = [Cluster.model_validate_json(line) for line in f]
        logger.info(f"Loaded {len(clusters)} clusters from {checkpoint_path}")
        return clusters
    except Exception as e:
        raise ValueError(f"Failed to load clusters from {checkpoint_path}: {e}")


def _build_cluster_tree(clusters: List[Cluster]) -> dict[str, ClusterTreeNode]:
    """Build a tree structure from a list of clusters.

    Args:
        clusters: List of clusters to build tree from

    Returns:
        Dictionary mapping cluster IDs to tree nodes
    """
    node_id_to_cluster = {}

    # Create tree nodes
    for cluster in clusters:
        node_id_to_cluster[cluster.id] = ClusterTreeNode(
            id=cluster.id,
            name=cluster.name,
            description=cluster.description,
            slug=cluster.slug,
            count=len(cluster.chat_ids),
            children=[],
        )

    # Link parent-child relationships
    for cluster in clusters:
        if cluster.parent_id and cluster.count > 0:
            node_id_to_cluster[cluster.parent_id].children.append(cluster.id)

    return node_id_to_cluster


def visualise_clusters(
    clusters: Optional[List[Cluster]] = None,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Print a hierarchical visualization of clusters to the terminal.

    This function loads clusters either from the provided list or from a checkpoint file,
    builds a tree representation, and prints it to the console.
    The visualization shows the hierarchical relationship between clusters
    with indentation and tree structure symbols.

    Args:
        clusters: List of clusters to visualize. If None, loads from checkpoint_path
        checkpoint_path: Path to checkpoint file to load clusters from

    Raises:
        ValueError: If neither clusters nor checkpoint_path is provided
        FileNotFoundError: If checkpoint file doesn't exist

    Example output:
        ╠══ Compare and improve Flutter and React state management (45 conversations)
        ║   ╚══ Improve and compare Flutter and React state management (32 conversations)
        ║       ╠══ Improve React TypeScript application (15 conversations)
        ║       ╚══ Compare and select Flutter state management solutions (17 conversations)
        ╠══ Optimize blog posts for SEO and improved user engagement (28 conversations)
    """
    # Load clusters
    if clusters is None:
        if checkpoint_path is None:
            raise ValueError("Either clusters or checkpoint_path must be provided")
        clusters = _load_clusters_from_checkpoint(checkpoint_path)

    logger.info(f"Visualizing {len(clusters)} clusters")

    # Build tree structure
    node_id_to_cluster = _build_cluster_tree(clusters)

    # Find root nodes and build the tree
    root_nodes = [node_id_to_cluster[cluster.id] for cluster in clusters if not cluster.parent_id]

    total_conversations = sum(node.count for node in root_nodes)
    fake_root = ClusterTreeNode(
        id="root",
        name="Clusters",
        description="All clusters",
        slug="all_clusters",
        count=total_conversations,
        children=[node.id for node in root_nodes],
    )

    tree_output = _build_tree_structure(fake_root, node_id_to_cluster, 0, False)
    # print(tree_output)
    return tree_output


def visualise_clusters_enhanced(
    clusters: Optional[List[Cluster]] = None,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Print an enhanced hierarchical visualization of clusters with colors and statistics.

    This function provides a more detailed visualization than visualise_clusters(),
    including conversation counts, percentages, progress bars, and descriptions.

    Args:
        clusters: List of clusters to visualize. If None, loads from checkpoint_path
        checkpoint_path: Path to checkpoint file to load clusters from

    Raises:
        ValueError: If neither clusters nor checkpoint_path is provided
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load clusters
    if clusters is None:
        if checkpoint_path is None:
            raise ValueError("Either clusters or checkpoint_path must be provided")
        clusters = _load_clusters_from_checkpoint(checkpoint_path)

    logger.info(f"Enhanced visualization of {len(clusters)} clusters")

    print("\n" + "=" * 80)
    print("🎯 ENHANCED CLUSTER VISUALIZATION")
    print("=" * 80)

    # Build tree structure
    node_id_to_cluster = _build_cluster_tree(clusters)

    # Calculate total conversations from root clusters only
    root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
    total_conversations = sum(len(cluster.chat_ids) for cluster in root_clusters)

    # Find root nodes
    root_nodes = [node_id_to_cluster[cluster.id] for cluster in root_clusters]

    fake_root = ClusterTreeNode(
        id="root",
        name=f"📚 All Clusters ({total_conversations:,} total conversations)",
        description="Hierarchical conversation clustering results",
        slug="all_clusters",
        count=total_conversations,
        children=[node.id for node in root_nodes],
    )

    tree_output = _build_enhanced_tree_structure(fake_root, node_id_to_cluster, 0, False, "", total_conversations)

    print(tree_output)

    # Add summary statistics
    print("=" * 80)
    print("📈 CLUSTER STATISTICS")
    print("=" * 80)
    print(f"📊 Total Clusters: {len(clusters)}")
    print(f"🌳 Root Clusters: {len(root_nodes)}")
    print(f"💬 Total Conversations: {total_conversations:,}")
    print(f"📏 Average Conversations per Root Cluster: {total_conversations / len(root_nodes):.1f}")
    print("=" * 80 + "\n")


# =============================================================================
# Convenience Functions for Integration with v1 Pipeline
# =============================================================================


def visualise_from_checkpoint_manager(
    checkpoint_manager,
    meta_cluster_model,
    *,
    style: str = "basic",
) -> None:
    """Visualize clusters using a CheckpointManager and meta cluster model.

    This function integrates with the v1 pipeline's CheckpointManager to automatically
    load and visualize clusters.

    Args:
        checkpoint_manager: CheckpointManager instance from v1 pipeline
        meta_cluster_model: Meta cluster model with checkpoint_filename
        style: Visualization style ("basic" or "enhanced")

    Raises:
        ValueError: If invalid style is provided
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not hasattr(meta_cluster_model, "checkpoint_filename"):
        raise ValueError("Meta cluster model must have checkpoint_filename attribute")

    checkpoint_path = checkpoint_manager.get_checkpoint_path(meta_cluster_model.checkpoint_filename)

    if style == "basic":
        visualise_clusters(checkpoint_path=checkpoint_path)
    elif style == "enhanced":
        visualise_clusters_enhanced(checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Invalid style '{style}'. Must be one of: basic, enhanced")


def visualise_pipeline_results(
    clusters: List[Cluster],
    *,
    style: Literal["basic", "enhanced"] = "enhanced",
) -> None:
    """Visualize clusters that are the result of a pipeline execution.

    Convenience function for visualizing clusters directly from pipeline results.

    Args:
        clusters: List of clusters from pipeline execution
        style: Visualization style ("basic" or "enhanced")

    Raises:
        ValueError: If invalid style is provided
    """
    if style == "basic":
        visualise_clusters(clusters)
    elif style == "enhanced":
        visualise_clusters_enhanced(clusters)
    else:
        raise ValueError(f"Invalid style '{style}'. Must be one of: basic, enhanced")
