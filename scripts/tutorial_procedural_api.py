import time
import asyncio
from contextlib import contextmanager


@contextmanager
def timer(message):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{message} took {end_time - start_time:.2f} seconds")


def show_section_header(title):
    """Display a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


with timer("Importing kura modules"):
    # Import the procedural Kura v1 components
    from kura import (
        summarise_conversations,
        generate_base_clusters_from_conversation_summaries,
        reduce_clusters_from_base_clusters,
        reduce_dimensionality_from_clusters,
        CheckpointManager,
    )

    # Import visualization functions

    # Import existing Kura models and types
    from kura.types import Conversation
    from kura.summarisation import SummaryModel
    from kura.cluster import ClusterModel
    from kura.meta_cluster import MetaClusterModel
    from kura.dimensionality import HDBUMAP

    # Import MiniBatch KMeans clustering method
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

with timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

# Save conversations to JSON for database loading
show_section_header("Saving Conversations")

with timer("Saving conversations to JSON"):
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

# Print conversation details
print("Sample Conversation Details:")
print(f"Chat ID: {sample_conversation.chat_id}")
print(f"Created At: {sample_conversation.created_at}")
print(f"Number of Messages: {len(sample_conversation.messages)}")
print()

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
    with timer("Conversation summarization"):
        summaries = await summarise_conversations(
            conversations, model=summary_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(summaries)} summaries")

    print("Step 2: Generating base clusters from summaries using MiniBatch KMeans...")
    with timer("MiniBatch KMeans clustering"):
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(clusters)} base clusters using MiniBatch KMeans")

    print("Step 3: Reducing clusters hierarchically...")
    with timer("Meta clustering"):
        reduced_clusters = await reduce_clusters_from_base_clusters(
            clusters, model=meta_cluster_model, checkpoint_manager=checkpoint_manager
        )
    print(f"Reduced to {len(reduced_clusters)} meta clusters")

    print("Step 4: Projecting clusters to 2D for visualization...")
    with timer("Dimensionality reduction"):
        projected_clusters = await reduce_dimensionality_from_clusters(
            reduced_clusters,
            model=dimensionality_model,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Generated {len(projected_clusters)} projected clusters")

    return reduced_clusters, projected_clusters


reduced_clusters, projected_clusters = asyncio.run(process_with_progress())
