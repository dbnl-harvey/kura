# Kura Documentation

## What is Kura?

> Kura is kindly sponsored by [Improving RAG](http://improvingrag.com). If you're wondering what goes on behind the scenes of any production RAG application, ImprovingRAG gives you a clear roadmap as to how to achieve it.

Kura is an open-source tool for understanding and visualizing chat data, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It helps you discover patterns, trends, and insights from user conversations by applying machine learning techniques to cluster similar interactions.

By iteratively summarizing and clustering conversations, Kura transforms raw chat data into actionable insights. It uses language models like Gemini to understand conversation content and organize them into meaningful hierarchical groups, helping you focus on the specific features to prioritize or issues to fix.

I've written a [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations) if you're interested in understanding the high level ideas.

## Quick Start

> Kura requires python 3.9 because of our dependency on UMAP.

```python
from kura import Kura
from kura.types import Conversation
import asyncio

# Initialize Kura with checkpoint directory
kura = Kura(
    checkpoint_dir="./tutorial_checkpoints"
)

# Load sample data (190 synthetic programming conversations)
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
print(f"Loaded {len(conversations)} conversations successfully!")

# Process conversations through the clustering pipeline
asyncio.run(kura.cluster_conversations(conversations))
# Expected output:
# Generated 190 summaries
# Generated 19 base clusters
# Reduced to 29 meta clusters
# Generated 29 projected clusters

# Visualize the results
kura.visualise_clusters()
# Expected output: Hierarchical tree showing 10 root clusters
# with topics like:
# - Create engaging, SEO-optimized content for online platforms (40 conversations)
# - Help me visualize and analyze data across platforms (30 conversations)
# - Troubleshoot and implement authentication in web APIs (22 conversations)
# ... and more
```

To explore more features, check out:
- [Installation Guide](getting-started/installation.md)
- [Quickstart Guide](getting-started/quickstart.md)
- [Core Concepts](core-concepts/overview.md)

## Technical Walkthrough

I've also recorded a technical deep dive into what Kura is and the ideas behind it if you'd rather watch than read.

<iframe width="560" height="315" src="https://www.youtube.com/embed/TPOP_jDiSVE?si=uvTond4LUwJGOn4F" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
