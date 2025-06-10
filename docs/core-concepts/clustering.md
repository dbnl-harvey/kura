# Clustering

Kura's clustering pipeline groups similar conversation summaries into meaningful clusters. This process is fundamental for large-scale analysis, enabling the discovery of dominant themes, understanding diverse user intents, and surfacing potentially "unknown unknown" patterns from vast quantities of conversational data. Clustering follows summarization and embedding in the Kura pipeline.

---

## Overview

**Clustering** in Kura organizes `ConversationSummary` objects (see [Summarization](summarization.md)) into groups based on semantic similarity. Each resulting cluster is assigned a descriptive name and a concise summary, making it easier to interpret the primary topics and user requests within the dataset. This bottom-up approach to pattern discovery is crucial for making sense of and navigating large collections of conversations.

- **Input:** A list of `ConversationSummary` objects (with or without embeddings)
- **Output:** A list of `Cluster` objects, each with a name, description, and associated conversation IDs

Clustering enables downstream tasks such as:
- Identifying and monitoring prevalent topics or user needs
- Visualizing trends and thematic structures in the data
- Facilitating efficient exploratory search and retrieval of related conversations
- Providing a foundation for hierarchical topic modeling through [Meta-Clustering](meta-clustering.md)

---

## The Clustering Model

Kura's main clustering logic is implemented in the `ClusterModel` class (see `kura/cluster.py`). This class orchestrates the embedding, grouping, and labeling of conversation summaries.

### Key Components

- **Clustering Method:** Determines how summaries are grouped (default: K-means, see `KmeansClusteringMethod`)
- **Embedding Model:** Used to convert summaries to vectors if not already embedded (default: `OpenAIEmbeddingModel`)
- **Cluster Naming:** Uses an LLM to generate a descriptive name and summary for each cluster, distinguishing it from others

#### Example: ClusterModel Initialization

```python
model = ClusterModel(
    clustering_method=KmeansClusteringMethod(),
    embedding_model=OpenAIEmbeddingModel(),
    max_concurrent_requests=50,
    model="openai/gpt-4o-mini",
)
```

---

## Clustering Pipeline

The clustering process consists of several steps:

1. **Embedding Summaries:**
   - If summaries do not already have embeddings, the model uses the configured embedding model to generate them.
   - Embedding is performed in batches and can be parallelized for efficiency.

   ```python
   embeddings = await self.embedding_model.embed([str(item) for item in summaries])
   ```

2. **Grouping Summaries:**
   - The clustering method (e.g., K-means) groups summaries based on their embeddings.
   - Each group is assigned a cluster ID.

   ```python
   cluster_id_to_summaries = self.clustering_method.cluster(items_with_embeddings)
   ```

3. **Generating Cluster Names and Descriptions:**
   - For each cluster, an LLM is prompted to generate a concise, two-sentence summary and a short, imperative cluster name.
   - The prompt includes both positive examples (summaries in the cluster) and contrastive examples (summaries from other clusters). Contrastive examples are crucial: they guide the LLM to produce highly specific and distinguishing names/descriptions, preventing overly generic labels and ensuring each cluster's unique essence is captured.

   ```python
   cluster = await self.generate_cluster(summaries, contrastive_examples)
   # Returns a Cluster object with name, description, and chat_ids
   ```

4. **Output:**
   - The result is a list of `Cluster` objects, each containing:
     - `name`: Imperative sentence capturing the main request/theme
     - `description`: Two-sentence summary of the cluster
     - `chat_ids`: List of conversation IDs in the cluster

---

## Cluster Naming and Description Generation

Cluster names and descriptions are generated using a large language model (LLM) with a carefully crafted prompt. The prompt:
- Instructs the LLM to summarize the group in two sentences (past tense)
- Requires the name to be an imperative sentence (e.g., "Help me debug Python code")
- Provides contrastive examples to ensure the name/summary is specific, distinct, and accurately reflects the cluster's content compared to others.
- Encourages specificity, especially for sensitive or harmful topics
- Reinforces privacy by instructing the LLM to avoid including any Personally Identifiable Information (PII) or proper nouns in the generated cluster names and descriptions, complementing the PII removal in the initial summarization phase.

**Prompt excerpt:**

```
Summarize all the statements into a clear, precise, two-sentence description in the past tense. ...
After creating the summary, generate a short name for the group of statements. This name should be at most ten words long ...
The cluster name should be a sentence in the imperative that captures the user's request. ...
```

---

## Configuration and Extensibility

- **Clustering Method:** Swap out `KmeansClusteringMethod` for other algorithms by implementing the `BaseClusteringMethod` interface.
- **Embedding Model:** Use any model implementing `BaseEmbeddingModel` (e.g., local or cloud-based embeddings).
- **LLM Model:** The LLM used for naming/describing clusters is configurable (default: `openai/gpt-4o-mini`).
- **Concurrency:** `max_concurrent_requests` controls parallelism for embedding and LLM calls.
- **Progress Reporting:** Optional integration with Rich or tqdm for progress bars and live cluster previews.

---

## Clustering Algorithms

Kura supports multiple clustering algorithms through the `BaseClusteringMethod` interface. Each algorithm has different characteristics and is suitable for different use cases.

### Standard K-means (`KmeansClusteringMethod`)

The default clustering method uses scikit-learn's standard K-means algorithm. This approach loads all embeddings into memory and computes centroids using the full dataset.

**Best for:**
- Small to medium datasets (< 50k conversations)
- When clustering accuracy is critical
- Reproducible results (deterministic with fixed random seed)
- Development and testing scenarios

**Characteristics:**
- **Memory usage:** High - loads entire embedding matrix into memory
- **Speed:** Moderate - full batch processing
- **Accuracy:** High - uses complete dataset for centroid calculation
- **Deterministic:** Yes - same results with same random seed

**Example usage:**
```python
from kura import ClusterModel, KmeansClusteringMethod

model = ClusterModel(
    clustering_method=KmeansClusteringMethod(clusters_per_group=10)
)
```

### MiniBatch K-means (`MiniBatchKmeansClusteringMethod`)

Optimized for large datasets, MiniBatch K-means processes data in small batches rather than loading everything into memory at once. This algorithm addresses the scalability bottlenecks identified in [Issue #92](https://github.com/567-labs/kura/issues/92).

**Best for:**
- Large datasets (100k+ conversations)
- Memory-constrained environments
- Production deployments with limited resources
- When clustering speed is more important than perfect accuracy

**Characteristics:**
- **Memory usage:** Low - processes data in configurable batch sizes
- **Speed:** Fast - incremental updates with early convergence
- **Accuracy:** Good - slightly less accurate due to stochastic nature
- **Deterministic:** No - results may vary between runs

**Configuration parameters:**
- `clusters_per_group`: Target items per cluster (default: 10)
- `batch_size`: Mini-batch size for processing (default: 1000)
- `max_iter`: Maximum iterations (default: 100)
- `random_state`: Random seed for reproducibility (default: 42)

**Example usage:**
```python
from kura import ClusterModel, MiniBatchKmeansClusteringMethod

# For large datasets
model = ClusterModel(
    clustering_method=MiniBatchKmeansClusteringMethod(
        clusters_per_group=15,
        batch_size=2000,  # Larger batches for better stability
        max_iter=150,
        random_state=42
    )
)
```

### HDBSCAN (`HDBSCANClusteringMethod`)

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) often provides superior results for conversation clustering tasks, especially when you don't know the optimal number of clusters in advance.

**Best for:**
- Exploratory analysis when you don't know how many topics exist
- Datasets with varying cluster densities (common and rare topics)
- Automatic noise and outlier detection
- When you need hierarchical cluster relationships
- Large datasets with high-dimensional embeddings

**Characteristics:**
- **Memory usage:** Moderate - more efficient than standard K-means
- **Speed:** Fast - good scalability for large datasets
- **Accuracy:** High - discovers natural cluster boundaries
- **Deterministic:** Yes - consistent results with same parameters

**Key advantages:**
- **Natural cluster discovery:** Automatically finds optimal number of clusters
- **Handles varying densities:** Can find clusters of different shapes and sizes
- **Noise detection:** Identifies outlier conversations that don't fit clear patterns
- **Hierarchical structure:** Creates tree of clusters showing topic relationships

**Configuration parameters:**
- `min_cluster_size`: Minimum conversations per cluster (default: 5)
- `min_samples`: Minimum samples for core points (default: 3)
- `cluster_selection_epsilon`: Distance threshold for merging (default: 0.0)
- `cluster_selection_method`: "eom" (Excess of Mass) or "leaf" (default: "eom")
- `metric`: Distance measure for embeddings (default: "euclidean")

**Example usage:**
```python
from kura import ClusterModel, HDBSCANClusteringMethod

# For exploratory analysis with automatic cluster discovery
model = ClusterModel(
    clustering_method=HDBSCANClusteringMethod(
        min_cluster_size=10,              # Minimum conversations per cluster
        min_samples=5,                    # Minimum samples for core points
        cluster_selection_epsilon=0.0,    # Distance threshold for merging
        cluster_selection_method="eom",   # Excess of Mass method
        metric="euclidean"                # Distance metric for embeddings
    )
)
```

### Algorithm Comparison

| Feature | K-means | MiniBatch K-means | HDBSCAN |
|---------|---------|-------------------|---------|
| **Dataset Size** | < 50k conversations | 100k+ conversations | Any size |
| **Memory Usage** | High (full matrix) | Low (batch-wise) | Moderate |
| **Processing Speed** | Moderate | Fast | Fast |
| **Clustering Quality** | High | Good | High |
| **Reproducibility** | Deterministic | Stochastic | Deterministic |
| **Cluster Count** | Fixed (calculated) | Fixed (calculated) | Automatic |
| **Noise Handling** | Forces all into clusters | Forces all into clusters | Identifies outliers |
| **Use Case** | Development, fixed groups | Production, large-scale | Exploratory, natural discovery |

### Memory Usage Comparison

For a dataset with 100,000 conversations using OpenAI embeddings (1536 dimensions):

- **Standard K-means:** ~1.2GB for embedding matrix alone
- **MiniBatch K-means:** ~12MB peak usage (with batch_size=1000)
- **HDBSCAN:** ~600MB for embedding matrix and cluster tree

### When to Use Each Algorithm

**Choose K-means when:**
- You want consistent, equal-sized clusters
- Dataset is small to medium (< 50k conversations)
- You need deterministic, reproducible results
- Working in development/testing environments

**Choose MiniBatch K-means when:**
- Experiencing memory issues with standard K-means
- Processing very large datasets (100k+ conversations)
- Speed is more important than perfect accuracy
- Working in resource-constrained production environments

**Choose HDBSCAN when:**
- You don't know how many clusters should exist
- Topics have naturally varying frequencies
- You want to identify and separate noise/outliers
- You need hierarchical relationships between clusters
- Doing exploratory analysis of conversation patterns

### Custom Clustering Methods

You can implement custom clustering algorithms by extending `BaseClusteringMethod`:

```python
from kura.base_classes import BaseClusteringMethod

class CustomClusteringMethod(BaseClusteringMethod):
    def cluster(self, items: list[dict]) -> dict[int, list]:
        # Your clustering implementation
        pass
```

---

## Hierarchical Analysis with Meta-Clustering

While the `ClusterModel` produces a flat list of semantically distinct clusters, Kura also supports the creation of hierarchical cluster structures through its **meta-clustering** capabilities (see [Meta-Clustering](meta-clustering.md)). This next step takes the output of the initial clustering (a list of `Cluster` objects) and groups these clusters into higher-level, more general parent clusters.

This hierarchical approach is particularly useful for:
- Managing and navigating a large number of base clusters.
- Discovering broader themes and relationships between groups of clusters.
- Enabling a multi-level exploratory search, from general topics down to specific conversation groups.

Refer to the [Meta-Clustering](meta-clustering.md) documentation for details on how Kura achieves this hierarchical organization.

---

## Output: Cluster Object

Each cluster is represented as a `Cluster` object (see `kura/types.py`):

```python
class Cluster(BaseModel):
    name: str
    description: str
    chat_ids: list[str]
    parent_id: Optional[int] = None
```

---

## Pipeline Integration

Clustering is the third major step in Kura's analysis pipeline:

1. **Loading:** Conversations are loaded
2. **Summarization:** Each conversation is summarized
3. **Embedding:** Summaries are embedded as vectors
4. **Clustering:** Embeddings are grouped into clusters (this step)
5. **Visualization/Analysis:** Clusters and summaries are explored

---

## References

- [Summarization](summarization.md)
- [Embedding](embedding.md)
- [API documentation](../api/index.md)
- [Source Code](https://github.com/567-labs/kura/blob/main/kura/cluster.py)
