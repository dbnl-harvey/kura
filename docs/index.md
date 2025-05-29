# Kura: Chat Data Analysis and Visualization

![Kura Architecture](assets/images/kura-architecture.png)

Kura is an open-source tool for understanding and visualizing chat data, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It helps you discover patterns, trends, and insights from user conversations by applying machine learning techniques to cluster similar interactions.

## Why Analyze Conversation Data?

As AI assistants and chatbots become increasingly central to product experiences, understanding how users interact with these systems at scale becomes a critical challenge. Manually reviewing thousands of conversations is impractical, yet crucial patterns and user needs often remain hidden in this data.

Kura addresses this challenge by:

- **Revealing user intent patterns** that may not be obvious from individual conversations
- **Identifying common user needs** to prioritize feature development
- **Discovering edge cases and failures** that require attention
- **Tracking usage trends** over time as your product evolves
- **Informing prompt engineering** by highlighting successful and problematic interactions

## Features

- **Conversation Summarization**: Automatically generate concise task descriptions from conversations
- **Hierarchical Clustering**: Group similar conversations at multiple levels of granularity
- **Interactive Visualization**: Explore clusters through map, tree, and detail views
- **Metadata Extraction**: Extract valuable context from conversations using LLMs
- **Custom Models**: Use your preferred embedding, summarization, and clustering methods
- **Web Interface**: Intuitive UI for exploring and analyzing conversation clusters
- **CLI Tools**: Command-line interface for scripting and automation
- **Checkpoint System**: Save and resume analysis sessions

## Installation

```bash
# Install from PyPI
pip install kura

# Or use uv for faster installation
uv pip install kura
```

## Quick Start

```python
from kura import Kura
from kura.types import Conversation
import asyncio

# Initialize Kura with default components
kura = Kura(
    checkpoint_dir="./tutorial_checkpoints"
)

# Load sample conversations from Hugging Face
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)

# Run the clustering pipeline
asyncio.run(kura.cluster_conversations(conversations))

# Visualize the results in the terminal
kura.visualise_clusters()

# Or start the web interface
# In your terminal: kura start-app --dir ./tutorial_checkpoints
# Access at http://localhost:8000
```

## Documentation

Explore the full documentation:

- **Getting Started**
  - [Installation Guide](getting-started/installation.md)
  - [Quickstart Guide](getting-started/quickstart.md)
  - [Configuration](getting-started/configuration.md)
  - [Tutorial: Procedural API](getting-started/tutorial-procedural-api.md)

- **Core Concepts**
  - [Overview](core-concepts/overview.md)
  - [Conversations](core-concepts/conversations.md)
  - [Embedding](core-concepts/embedding.md)
  - [Clustering](core-concepts/clustering.md)
  - [Summarization](core-concepts/summarization.md)
  - [Meta-Clustering](core-concepts/meta-clustering.md)
  - [Dimensionality Reduction](core-concepts/dimensionality-reduction.md)

- **API Reference**
  - [API Documentation](api/index.md)

- **Advanced Topics**
  - [Issue Tracker Integration](itracker.md)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
