import time
import asyncio
from contextlib import contextmanager


class TimerManager:
    """A timer class that collects timing data for review."""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def timer(self, message):
        """Context manager to time operations and store results."""
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        self.timings[message] = duration
        print(f"{message} took {duration:.2f} seconds")

    def print_summary(self):
        """Print a summary of all collected timings."""
        print(f"\n{'=' * 60}")
        print(f"{'TIMING SUMMARY':^60}")
        print(f"{'=' * 60}")

        total_time = sum(self.timings.values())

        for operation, duration in self.timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{operation:<40} {duration:>8.2f}s ({percentage:>5.1f}%)")

        print(f"{'-' * 60}")
        print(f"{'Total Time':<40} {total_time:>8.2f}s")
        print(f"{'=' * 60}\n")


# Create a global timer manager instance
timer_manager = TimerManager()


def show_section_header(title):
    """Display a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


with timer_manager.timer("Importing kura modules"):
    # Import the procedural Kura v1 components
    from kura import (
        summarise_conversations,
        generate_base_clusters_from_conversation_summaries,
        reduce_clusters_from_base_clusters,
        reduce_dimensionality_from_clusters,
        CheckpointManager,
    )

    # Import Parquet checkpoint manager if available
    try:
        from kura import ParquetCheckpointManager

        PARQUET_AVAILABLE = True
    except ImportError:
        ParquetCheckpointManager = None
        PARQUET_AVAILABLE = False

    # Import visualization functions

    # Import existing Kura models and types
    from kura.types import Conversation
    from kura.summarisation import SummaryModel
    from kura.cluster import ClusterModel
    from kura.meta_cluster import MetaClusterModel
    from kura.dimensionality import HDBUMAP

    # Import all available clustering methods
    from kura.k_means import MiniBatchKmeansClusteringMethod

    from rich.console import Console


# Set up individual models
console = Console()
summary_model = SummaryModel(console=console)

# Initialize MiniBatch KMeans clustering method with appropriate parameters
minibatch_kmeans_clustering = MiniBatchKmeansClusteringMethod(
    clusters_per_group=10,  # Target items per cluster
    batch_size=1000,  # Mini-batch size for processing
    max_iter=100,  # Maximum iterations
    random_state=42,  # Random seed for reproducibility
)

# Use MiniBatch KMeans clustering method in ClusterModel
cluster_model = ClusterModel(
    clustering_method=minibatch_kmeans_clustering, console=console
)
meta_cluster_model = MetaClusterModel(console=console)
dimensionality_model = HDBUMAP()

# Set up checkpointing
checkpoint_manager = CheckpointManager("./tutorial_checkpoints", enabled=True)

with timer_manager.timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

# Save conversations to JSON for database loading
show_section_header("Saving Conversations")

with timer_manager.timer("Saving conversations to JSON"):
    import json
    import os

    # Ensure checkpoint directory exists
    os.makedirs("./tutorial_checkpoints", exist_ok=True)

    # Convert conversations to JSON format
    conversations_data = [conv.model_dump() for conv in conversations]

    # Save to conversations.json
    with open("./tutorial_checkpoints/conversations.json", "w") as f:
        json.dump(conversations_data, f, indent=2, default=str)

print(
    f"Saved {len(conversations)} conversations to tutorial_checkpoints/conversations.json\n"
)

# Sample conversation examination
show_section_header("Sample Data Examination")

sample_conversation = conversations[0]

# Sample messages
print("Sample Messages:")
for i, msg in enumerate(sample_conversation.messages[:3]):
    content_preview = (
        msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
    )
    print(f"  {msg.role}: {content_preview}")

print()

# Processing section
show_section_header("Conversation Processing with MiniBatch KMeans")

print("Starting conversation clustering with MiniBatch KMeans...")


async def process_with_progress():
    """Process conversations step by step using the procedural API with MiniBatch KMeans clustering."""
    print("Step 1: Generating conversation summaries...")
    with timer_manager.timer("Conversation summarization"):
        summaries = await summarise_conversations(
            conversations, model=summary_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(summaries)} summaries")

    print("Step 2: Generating base clusters from summaries using MiniBatch KMeans...")
    with timer_manager.timer("MiniBatch KMeans clustering"):
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(clusters)} base clusters using MiniBatch KMeans")

    print("Step 3: Reducing clusters hierarchically...")
    with timer_manager.timer("Meta clustering"):
        reduced_clusters = await reduce_clusters_from_base_clusters(
            clusters, model=meta_cluster_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Reduced to {len(reduced_clusters)} meta clusters")

    print("Step 4: Projecting clusters to 2D for visualization...")
    with timer_manager.timer("Dimensionality reduction"):
        projected_clusters = await reduce_dimensionality_from_clusters(
            reduced_clusters,
            model=dimensionality_model,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Generated {len(projected_clusters)} projected clusters")

    return reduced_clusters, projected_clusters


reduced_clusters, projected_clusters = asyncio.run(process_with_progress())

# Parquet format demonstration (if available)
if PARQUET_AVAILABLE:
    show_section_header("Parquet Format Demonstration")

    print("Demonstrating Parquet checkpoint format for efficient storage...")
    print(
        "This format offers significant space savings and faster loading for large datasets.\n"
    )

    # Set up Parquet checkpoint manager
    parquet_checkpoint_manager = ParquetCheckpointManager(
        "./tutorial_parquet_checkpoints", enabled=True
    )

    async def process_with_parquet():
        """Process conversations using Parquet checkpoints."""
        print("Step 1: Generating summaries with Parquet checkpoints...")
        with timer_manager.timer("Parquet summarization"):
            summaries = await summarise_conversations(
                conversations,
                model=summary_model,
                checkpoint_manager=parquet_checkpoint_manager,
            )
        print(f"Generated {len(summaries)} summaries using Parquet format")

        print("Step 2: Generating clusters with Parquet checkpoints...")
        with timer_manager.timer("Parquet clustering"):
            clusters = await generate_base_clusters_from_conversation_summaries(
                summaries,
                model=cluster_model,
                checkpoint_manager=parquet_checkpoint_manager,
            )
        print(f"Generated {len(clusters)} clusters using Parquet format")

        print("Step 3: Meta clustering with Parquet checkpoints...")
        with timer_manager.timer("Parquet meta clustering"):
            reduced_clusters = await reduce_clusters_from_base_clusters(
                clusters,
                model=meta_cluster_model,
                checkpoint_manager=parquet_checkpoint_manager,
            )
        print(f"Reduced to {len(reduced_clusters)} meta clusters using Parquet format")

        print("Step 4: Dimensionality reduction with Parquet checkpoints...")
        with timer_manager.timer("Parquet dimensionality reduction"):
            projected_clusters = await reduce_dimensionality_from_clusters(
                reduced_clusters,
                model=dimensionality_model,
                checkpoint_manager=parquet_checkpoint_manager,
            )
        print(
            f"Generated {len(projected_clusters)} projected clusters using Parquet format"
        )

        return reduced_clusters, projected_clusters

    # Run with Parquet
    parquet_reduced_clusters, parquet_projected_clusters = asyncio.run(
        process_with_parquet()
    )

    # Compare file sizes
    show_section_header("Format Comparison")

    import os

    def get_directory_size(directory):
        """Get total size of all files in a directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def format_size(size_bytes):
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.2f} {units[unit_index]}"

    # Calculate sizes
    jsonl_size = get_directory_size("./tutorial_checkpoints")
    parquet_size = get_directory_size("./tutorial_parquet_checkpoints")

    print("File Size Comparison:")
    print(f"JSONL format:    {format_size(jsonl_size)}")
    print(f"Parquet format:  {format_size(parquet_size)}")

    if jsonl_size > 0 and parquet_size > 0:
        reduction = (jsonl_size - parquet_size) / jsonl_size * 100
        compression_ratio = jsonl_size / parquet_size
        space_saved = jsonl_size - parquet_size

        print("\nSpace Savings:")
        print(f"• {reduction:.1f}% reduction in file size")
        print(f"• {compression_ratio:.1f}x compression ratio")
        print(f"• {format_size(space_saved)} space saved")

        print("\nBenefits of Parquet format:")
        print("• Columnar storage ideal for embeddings and numerical data")
        print("• Built-in compression (using snappy by default)")
        print("• Faster loading for analytical workloads")
        print("• Self-describing schema")
        print("• Compatible with data science tools (pandas, polars, etc.)")

    print("\nTo use Parquet checkpoints in your code:")
    print("```python")
    print("from kura import ParquetCheckpointManager")
    print("")
    print("# Create Parquet checkpoint manager")
    print(
        "checkpoint_manager = ParquetCheckpointManager('./checkpoints', enabled=True)"
    )
    print("")
    print("# Use in pipeline functions")
    print("summaries = await summarise_conversations(")
    print(
        "    conversations, model=summary_model, checkpoint_manager=checkpoint_manager"
    )
    print(")")
    print("```")

else:
    show_section_header("Parquet Format (Not Available)")
    print("PyArrow is not installed. To use Parquet checkpoints, install it with:")
    print("pip install pyarrow")
    print("\nParquet format offers:")
    print("• Significant file size reduction (typically 50-80% smaller)")
    print("• Faster loading for large datasets")
    print("• Better compression for numerical data like embeddings")
    print("• Compatibility with data science ecosystem")
