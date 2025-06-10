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

## Clustering Methods: K-Means vs HDBSCAN

While K-means clustering is the default method in Kura, **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** often provides superior results for conversation clustering tasks. Understanding when and why to use each method can significantly improve your clustering outcomes.

### Why HDBSCAN is Often Better for Conversations

**1. Natural Cluster Discovery**
- **K-means** in Kura calculates the number of clusters based on a target group size (conversations per cluster), which may not reflect natural topic boundaries
- **HDBSCAN** automatically discovers the optimal number of clusters based on data density, making it ideal for exploratory analysis

**2. Handles Varying Cluster Densities**
- **K-means** assumes clusters are spherical and roughly equal in size, which rarely matches real conversation patterns
- **HDBSCAN** can find clusters of different shapes and densities, better reflecting how conversational topics naturally group

**3. Automatic Noise Detection**
- **K-means** forces every conversation into a cluster, even outliers and noise
- **HDBSCAN** identifies and separates outlier conversations that don't belong to any clear topic, improving cluster quality

**4. Hierarchical Structure**
- **K-means** produces flat clusters only
- **HDBSCAN** naturally creates a hierarchical tree of clusters, showing how topics relate and can be subdivided

### When to Use Each Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|---------|
| **Exploratory Analysis** | HDBSCAN | Don't know how many topics exist |
| **Target Group Size** | K-means | When you want roughly equal-sized clusters |
| **Mixed Topic Densities** | HDBSCAN | Some topics are common, others rare |
| **Noise/Outlier Handling** | HDBSCAN | Want to identify conversations that don't fit clear patterns |
| **Large Datasets** | HDBSCAN | Better scalability for high-dimensional embedding spaces |
| **Hierarchical Analysis** | HDBSCAN | Need to understand topic relationships and sub-topics |

### Using HDBSCAN in Kura

To use HDBSCAN instead of K-means, initialize your `ClusterModel` with the `HDBSCANClusteringMethod`:

```python
from kura.hdbscan import HDBSCANClusteringMethod
from kura.cluster import ClusterModel

# Configure HDBSCAN clustering
hdbscan_clustering = HDBSCANClusteringMethod(
    min_cluster_size=10,              # Minimum conversations per cluster
    min_samples=5,                    # Minimum samples for core points
    cluster_selection_epsilon=0.0,    # Distance threshold for merging
    cluster_selection_method="eom",   # Excess of Mass method
    metric="euclidean"                # Distance metric for embeddings
)

# Create cluster model with HDBSCAN
cluster_model = ClusterModel(
    clustering_method=hdbscan_clustering,
    embedding_model=OpenAIEmbeddingModel(),
    max_concurrent_requests=50,
    model="openai/gpt-4o-mini",
)
```

### Key HDBSCAN Parameters

- **`min_cluster_size`**: Minimum number of conversations required to form a cluster. Larger values create fewer, more substantial clusters
- **`min_samples`**: Number of conversations in a neighborhood for a point to be considered a core point. Lower values allow more fine-grained clustering
- **`cluster_selection_method`**:
  - `"eom"` (Excess of Mass): Better for finding clusters of varying densities
  - `"leaf"`: Better when you want many small, tight clusters
- **`metric`**: Distance measure for embeddings. `"euclidean"` works well for normalized text embeddings

### Example Results Comparison

For a dataset of 500 customer support conversations:

**K-means (k=10)**:
- Forces exactly 10 clusters
- May group unrelated edge cases together
- Equal-sized clusters regardless of actual topic frequency

**HDBSCAN (min_cluster_size=15)**:
- Discovers 8 meaningful clusters automatically
- Identifies 23 outlier conversations as noise
- Clusters vary in size based on topic popularity
- Shows how "billing issues" splits into "payment failures" and "invoice questions"

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
