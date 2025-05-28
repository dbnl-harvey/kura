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
    # Import the class-based Kura API
    from kura import Kura
    from kura.types import Conversation
    import json
    import os


# Initialize Kura with checkpoint directory
kura = Kura(checkpoint_dir="./tutorial_checkpoints_class")

with timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

# Save conversations to JSON for database loading
show_section_header("Saving Conversations")

with timer("Saving conversations to JSON"):
    # Ensure checkpoint directory exists
    os.makedirs("./tutorial_checkpoints_class", exist_ok=True)

    # Convert conversations to JSON format
    conversations_data = [conv.model_dump() for conv in conversations]

    # Save to conversations.json
    with open("./tutorial_checkpoints_class/conversations.json", "w") as f:
        json.dump(conversations_data, f, indent=2, default=str)

print(
    f"Saved {len(conversations)} conversations to tutorial_checkpoints_class/conversations.json\n"
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
show_section_header("Conversation Processing")

print("Starting conversation clustering...")


async def process_conversations():
    """Process conversations using the class-based API."""
    print("Running the complete pipeline through Kura class...")
    with timer("Complete pipeline processing"):
        result = await kura.cluster_conversations(conversations)

    print("\nPipeline complete!")
    print("Processing Summary:")
    print(f"  • Input conversations: {len(conversations)}")
    print(f"  • Result clusters: {len(result) if result else 0}")
    print(f"  • Checkpoints saved to: {kura.checkpoint_dir}")

    return result


# Run the processing
result = asyncio.run(process_conversations())

print()

# Visualization section
show_section_header("Cluster Visualization")

print("1. Basic cluster visualization:")
print("-" * 50)
with timer("Basic visualization"):
    kura.visualise_clusters()

print("\n2. Enhanced cluster visualization:")
print("-" * 50)
with timer("Enhanced visualization"):
    kura.visualise_clusters_enhanced()

print("\n3. Rich cluster visualization:")
print("-" * 50)
with timer("Rich visualization"):
    kura.visualise_clusters_rich()

# Summary statistics
show_section_header("Cluster Statistics")

if result:
    # Count root clusters (clusters with no parent)
    root_clusters = [c for c in result if c.parent_id is None]
    print(f"Total Clusters: {len(result)}")
    print(f"Root Clusters: {len(root_clusters)}")
    print(f"Total Conversations: {sum(c.count for c in root_clusters)}")
    print(
        f"Average Conversations per Root Cluster: {sum(c.count for c in root_clusters) / len(root_clusters):.1f}"
    )

    # Show cluster size distribution
    print("\nCluster Size Distribution:")
    large_clusters = [c for c in root_clusters if c.count > 100]
    medium_clusters = [c for c in root_clusters if 21 <= c.count <= 100]
    small_clusters = [c for c in root_clusters if 6 <= c.count <= 20]
    tiny_clusters = [c for c in root_clusters if 1 <= c.count <= 5]

    print(f"  • Large clusters (>100 conversations): {len(large_clusters)}")
    print(f"  • Medium clusters (21-100 conversations): {len(medium_clusters)}")
    print(f"  • Small clusters (6-20 conversations): {len(small_clusters)}")
    print(f"  • Tiny clusters (1-5 conversations): {len(tiny_clusters)}")
else:
    print("No clusters were generated.")

print("\n" + "=" * 80)
print("✨ TUTORIAL COMPLETE!")
print("=" * 80)

print("Class-Based API Benefits Demonstrated:")
print("  ✅ Simple one-line processing with cluster_conversations()")
print("  ✅ Automatic checkpoint management")
print("  ✅ Built-in visualization methods")
print("  • All components integrated in single class")
print("  • Direct access to intermediate results")
print("  • Multiple visualization styles available")
print()

print("Key Differences from Procedural API:")
print("  • Single orchestrating class instead of separate functions")
print("  • State management within the Kura instance")
print("  • Less granular control but simpler to use")
print("  • All models initialized automatically")
print()

print(f"Check '{kura.checkpoint_dir}' for saved intermediate results!")
print("Try different visualization styles:")
print("  - kura.visualise_clusters() for basic view")
print("  - kura.visualise_clusters_enhanced() for enhanced view")
print("  - kura.visualise_clusters_rich() for rich view")
print("\nTo start the web interface, run:")
print(f"  kura start-app --dir {kura.checkpoint_dir}")
