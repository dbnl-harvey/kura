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
- [kura/cluster.py](../../kura/cluster.py)
