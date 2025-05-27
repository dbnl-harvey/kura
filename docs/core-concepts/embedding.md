# Embedding

Kura's embedding pipeline transforms text (such as conversation summaries) into high-dimensional vector representations. These embeddings are essential for downstream tasks like clustering, search, and visualization, enabling Kura to analyze and organize large volumes of conversational data.

---

## Overview

**Embedding** in Kura refers to the process of converting text into numerical vectors (embeddings) that capture semantic meaning. These vectors allow for efficient similarity search, clustering, and visualization of conversations and summaries.

- **Input:** A list of texts (e.g., conversation summaries, messages, or cluster descriptions)
- **Output:** A list of vector embeddings (`list[list[float]]`), typically one per input text

---

## The Embedding Model

Kura uses an `EmbeddingModel` (see `kura/embedding.py`) that implements the `BaseEmbeddingModel` interface. Multiple backends are supported:

- **OpenAIEmbeddingModel**: Uses OpenAI's API (e.g., `text-embedding-3-small`) for high-quality embeddings
- **SentenceTransformerEmbeddingModel**: Uses local models from the `sentence-transformers` library (e.g., `all-MiniLM-L6-v2`)

All embedding models must implement the following interface (see `kura/base_classes/embedding.py`):

```python
class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of lists of floats"""
        pass
```

### Key Features

- **Batching:** Texts are automatically split into batches for efficient processing
- **Concurrency:** Multiple batches are embedded in parallel (configurable concurrency)
- **Retry Logic:** Embedding requests are retried on failure for robustness
- **Extensibility:** New embedding backends can be added by subclassing `BaseEmbeddingModel`
- **Checkpointing:** Embeddings can be cached as part of the pipeline to avoid recomputation

---

## Output: Embeddings

The result of embedding is a list of vectors, each representing an input text. Embeddings are typically attached to summaries or clusters for downstream analysis.

Example output for a batch of texts:

```python
embeddings = await embedding_model.embed([
    "Summarize the user's request.",
    "Cluster similar conversations together."
])
# embeddings: list[list[float]]
```

When used in the pipeline, embeddings are stored in objects such as `ConversationSummary`:

```python
class ConversationSummary(BaseModel):
    chat_id: str
    summary: str
    ...
    embedding: Optional[list[float]] = None
```

- **embedding**: The vector representation of the summary (or other text)

---

## Pipeline Integration

Embedding is a core step in Kura's analysis pipeline:

1. **Loading**: Conversations are loaded from various sources
2. **Summarization**: Each conversation is summarized
3. **Embedding**: Summaries (or other texts) are embedded as vectors
4. **Clustering**: Embeddings are grouped into clusters
5. **Visualization/Analysis**: Clusters and embeddings are explored

---

## Embeddable Object Representations

All major objects that need to be embedded in Kura (such as `ConversationSummary`, `Cluster`, and `ProjectedCluster`) implement `__str__` methods. This ensures that each object can be converted to a meaningful text representation before embedding.

- **Requirement:** Any object passed to an embedding model must provide a `__str__` method that captures its semantic content.
- **Examples:**
  - `ConversationSummary` uses a custom `__str__` to include summary, request, task, and other fields in a structured format.
  - `Cluster` and `ProjectedCluster` use `__str__` to return their name and description.

This design allows embedding models to work generically with a variety of object types, as long as they implement a suitable `__str__` method.

---

## References

- [API documentation](../api/index.md)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [OpenAI Embeddings documentation](https://platform.openai.com/docs/guides/embeddings)

---

## TODO: Additional Embedding Providers

- Support for other embedding providers (e.g., Cohere, HuggingFace Inference API, Google Vertex AI, local GPU models)
- Community contributions and suggestions are welcome!
