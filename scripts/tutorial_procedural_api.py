import time
import asyncio
import argparse
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


def parse_arguments():
    """Parse command-line arguments for clustering configuration."""
    parser = argparse.ArgumentParser(
        description="Tutorial for Kura's procedural API with configurable clustering methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Clustering method selection
    parser.add_argument(
        "--clustering-method",
        choices=["kmeans", "hdbscan"],
        default="hdbscan",
        help="Clustering method to use",
    )

    # HDBSCAN parameters
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN clustering",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples for core points in HDBSCAN",
    )
    parser.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default="eom",
        help="HDBSCAN cluster selection method",
    )

    # K-means parameters
    parser.add_argument(
        "--clusters-per-group",
        type=int,
        default=15,
        help="Target conversations per cluster for K-means",
    )

    # General parameters
    parser.add_argument(
        "--dataset",
        default="ivanleomk/synthetic-gemini-conversations",
        help="Hugging Face dataset to use",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--checkpoint-dir",
        default="./tutorial_checkpoints",
        help="Directory for storing checkpoints",
    )
    parser.add_argument(
        "--visualization-style",
        choices=["basic", "enhanced", "rich"],
        default="enhanced",
        help="Visualization style for results",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip the visualization demonstration",
    )

    return parser.parse_args()


def create_clustering_method(args):
    """Create the appropriate clustering method based on arguments."""
    if args.clustering_method == "hdbscan":
        from kura.hdbscan import HDBSCANClusteringMethod

        print("🔬 Using HDBSCAN clustering with:")
        print(f"   • min_cluster_size: {args.min_cluster_size}")
        print(f"   • min_samples: {args.min_samples}")
        print(f"   • cluster_selection_method: {args.cluster_selection_method}")
        print("   • metric: euclidean")

        return HDBSCANClusteringMethod(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=0.0,
            cluster_selection_method=args.cluster_selection_method,
            metric="euclidean",
        )

    elif args.clustering_method == "kmeans":
        from kura.k_means import KmeansClusteringMethod

        print("🔬 Using K-means clustering with:")
        print(f"   • clusters_per_group: {args.clusters_per_group}")

        return KmeansClusteringMethod(clusters_per_group=args.clusters_per_group)

    else:
        raise ValueError(f"Unknown clustering method: {args.clustering_method}")


def main():
    """Main function to run the configurable tutorial."""
    args = parse_arguments()

    # Display configuration
    print("🚀 Kura Procedural API Tutorial")
    print("=" * 60)
    print(f"📊 Clustering Method: {args.clustering_method.upper()}")
    print(f"📁 Dataset: {args.dataset} ({args.split})")
    print(f"💾 Checkpoint Directory: {args.checkpoint_dir}")
    print(f"🎨 Visualization Style: {args.visualization_style}")
    print("=" * 60)

    # Run the async tutorial
    asyncio.run(run_tutorial(args))


async def run_tutorial(args):
    """Run the tutorial with the specified configuration."""
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
        from kura.visualization import (
            visualise_clusters_rich,
            visualise_from_checkpoint_manager,
            visualise_pipeline_results,
        )

        # Import existing Kura models and types
        from kura.types import Conversation
        from kura.summarisation import SummaryModel
        from kura.cluster import ClusterModel
        from kura.meta_cluster import MetaClusterModel
        from kura.dimensionality import HDBUMAP

        from rich.console import Console

    # Set up individual models
    console = Console()
    summary_model = SummaryModel(console=console)

    # Create clustering method based on arguments
    clustering_method = create_clustering_method(args)

    # Use selected clustering method in ClusterModel
    cluster_model = ClusterModel(clustering_method=clustering_method, console=console)

    meta_cluster_model = MetaClusterModel(console=console)
    dimensionality_model = HDBUMAP()

    # Set up checkpointing
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, enabled=True)

    with timer("Loading sample conversations"):
        conversations = Conversation.from_hf_dataset(args.dataset, split=args.split)

    print(f"Loaded {len(conversations)} conversations successfully!\n")

    # Save conversations to JSON for database loading
    show_section_header("Saving Conversations")

    with timer("Saving conversations to JSON"):
        import json
        import os

        # Ensure checkpoint directory exists
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Convert conversations to JSON format
        conversations_data = [conv.model_dump() for conv in conversations]

        # Save to conversations.json
        with open(f"{args.checkpoint_dir}/conversations.json", "w") as f:
            json.dump(conversations_data, f, indent=2, default=str)

    print(
        f"Saved {len(conversations)} conversations to {args.checkpoint_dir}/conversations.json\n"
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
    clustering_method_name = args.clustering_method.upper()
    show_section_header(f"Conversation Processing with {clustering_method_name}")

    print(f"Starting conversation clustering with {clustering_method_name}...")

    async def process_with_progress():
        """Process conversations step by step using the procedural API."""
        print("Step 1: Generating conversation summaries...")
        with timer("Conversation summarization"):
            summaries = await summarise_conversations(
                conversations,
                model=summary_model,
                checkpoint_manager=checkpoint_manager,
            )
        print(f"Generated {len(summaries)} summaries")

        print(
            f"Step 2: Generating base clusters from summaries using {clustering_method_name}..."
        )
        with timer(f"{clustering_method_name} clustering"):
            clusters = await generate_base_clusters_from_conversation_summaries(
                summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
            )
        print(f"Generated {len(clusters)} base clusters using {clustering_method_name}")

        print("Step 3: Reducing clusters hierarchically...")
        with timer("Meta clustering"):
            reduced_clusters = await reduce_clusters_from_base_clusters(
                clusters,
                model=meta_cluster_model,
                checkpoint_manager=checkpoint_manager,
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

    reduced_clusters, projected_clusters = await process_with_progress()

    print(
        f"\nPipeline complete! Generated {len(projected_clusters)} projected clusters!\n"
    )

    print("Processing Summary:")
    print(f"  • Input conversations: {len(conversations)}")
    print(f"  • Final reduced clusters: {len(reduced_clusters)}")
    print(f"  • Final projected clusters: {len(projected_clusters)}")
    print(f"  • Checkpoints saved to: {checkpoint_manager.checkpoint_dir}")
    print()

    # Skip visualization if requested
    if args.skip_visualization:
        print(
            "⏭️  Skipping visualization demonstration (--skip-visualization flag used)"
        )
        print_summary(args, clustering_method_name, checkpoint_manager)
        return

    print("=" * 80)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 80)

    print(f"\n1. {args.visualization_style.title()} cluster visualization:")
    print("-" * 50)
    with timer(f"{args.visualization_style.title()} visualization"):
        if args.visualization_style == "basic":
            visualise_from_checkpoint_manager(
                checkpoint_manager, meta_cluster_model, style="basic"
            )
        elif args.visualization_style == "enhanced":
            visualise_pipeline_results(reduced_clusters, style="enhanced")
        elif args.visualization_style == "rich":
            visualise_clusters_rich(reduced_clusters, console=console)

    # Show comparison with other styles if not using basic
    if args.visualization_style != "basic":
        print("\n2. Basic cluster visualization (for comparison):")
        print("-" * 50)
        with timer("Basic visualization"):
            visualise_from_checkpoint_manager(
                checkpoint_manager, meta_cluster_model, style="basic"
            )

    print_summary(args, clustering_method_name, checkpoint_manager)


def print_summary(args, clustering_method_name, checkpoint_manager):
    """Print the final summary with tips and recommendations."""
    print("=" * 80)
    print("✨ TUTORIAL COMPLETE!")
    print("=" * 80)

    print("Procedural API Benefits Demonstrated:")
    print("  ✅ Step-by-step processing with individual control")
    print("  ✅ Flexible checkpoint management")
    print("  • Clear separation of concerns")
    print("  • Easy to customize individual steps")
    print("  • Multiple visualization options")
    print()

    print(f"{clustering_method_name} Clustering Features Demonstrated:")
    if args.clustering_method == "hdbscan":
        print("  • Density-based clustering for natural groupings")
        print("  • Automatic noise detection and outlier handling")
        print("  • No need to specify number of clusters in advance")
        print("  • Better handling of clusters with varying densities")
        print("  • Hierarchical cluster tree structure")
    elif args.clustering_method == "kmeans":
        print("  • Fast and deterministic clustering")
        print("  • Predictable cluster sizes based on target group size")
        print("  • Good for when you want roughly equal-sized groups")
        print("  • Simple parameter tuning")
    print()

    print("Visualization Features Demonstrated:")
    print("  • Basic hierarchical tree view")
    print("  • Enhanced view with statistics and progress bars")
    print("  • Rich-formatted output with colors and tables")
    print("  • Direct checkpoint integration")
    print("  • Pipeline result visualization")
    print()

    print("CheckpointManager Integration:")
    print("  • Automatic checkpoint loading and saving")
    print("  • Seamless integration with visualization functions")
    print("  • Resume processing from any checkpoint")
    print("  • Visualize results without re-running pipeline")
    print()

    print(f"💾 Results saved to: {checkpoint_manager.checkpoint_dir}")
    print()
    print("🔄 Try different configurations:")
    if args.clustering_method == "hdbscan":
        print("  python scripts/tutorial_procedural_api.py --clustering-method kmeans")
        print(
            "  python scripts/tutorial_procedural_api.py --min-cluster-size 10 --visualization-style rich"
        )
    else:
        print("  python scripts/tutorial_procedural_api.py --clustering-method hdbscan")
        print(
            "  python scripts/tutorial_procedural_api.py --clusters-per-group 20 --visualization-style rich"
        )

    print("\n🎨 Try different visualization styles:")
    print("  python scripts/tutorial_procedural_api.py --visualization-style basic")
    print("  python scripts/tutorial_procedural_api.py --visualization-style enhanced")
    print("  python scripts/tutorial_procedural_api.py --visualization-style rich")

    print("\n📊 Compare clustering methods:")
    print(
        "  python scripts/tutorial_procedural_api.py --clustering-method kmeans --checkpoint-dir ./kmeans_checkpoints"
    )
    print(
        "  python scripts/tutorial_procedural_api.py --clustering-method hdbscan --checkpoint-dir ./hdbscan_checkpoints"
    )


if __name__ == "__main__":
    main()
