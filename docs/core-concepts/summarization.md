# Summarization

Kura's summarization pipeline is designed to extract concise, structured, and privacy-preserving summaries from conversations between users and AI assistants. This process is central to Kura's ability to analyze, cluster, and visualize large volumes of conversational data.

---

## Overview

**Summarization** in Kura transforms each conversation into a structured summary, capturing the user's intent, the main task, languages involved, safety concerns, user frustration, and any assistant errors. This enables downstream analysis such as clustering, search, and visualization.

- **Input:** A `Conversation` object (see [Conversations](conversations.md)), containing:
  - `chat_id`: Unique identifier
  - `created_at`: Timestamp
  - `messages`: List of messages (each with `role`, `content`, `created_at`)
  - `metadata`: Optional dictionary of extra info
- **Output:** A `ConversationSummary` object (see below)

---

## The Summarization Model

Kura uses a `SummaryModel` (see `kura/summarisation.py`) that implements the `BaseSummaryModel` interface. The default model is based on large language models (LLMs) such as OpenAI's GPT-4o, but the interface supports other backends as well.

### Key Features

- **Concurrency:** Summarization is performed in parallel for efficiency.
- **Hooks/Extractors:** Optional extractors can add custom metadata to each summary.
- **Checkpointing:** Summaries can be cached and reloaded to avoid recomputation.

---

## Summarization Prompt

The summarization model uses a carefully crafted prompt to extract the following fields from each conversation:

1. **Summary**: A clear, concise summary (max 2 sentences, no PII or proper nouns)
2. **Request**: The user's overall request, starting with "The user's overall request for the assistant is to ..."
3. **Languages**: Main human and programming languages present
4. **Task**: The main task, starting with "The task is to ..."
5. **Concerning Score**: Safety concern rating (1–5)
6. **User Frustration**: User frustration rating (1–5)
7. **Assistant Errors**: List of errors made by the assistant

**Prompt excerpt:**

```
Your job is to extract key information from this conversation. Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; specificity around potentially harmful conversations is necessary for effective monitoring.

When extracting information, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addresses, and so on. Do not include any proper nouns.

Extract the following information:

1. **Summary**: ...
2. **Request**: ...
3. **Languages**: ...
4. **Task**: ...
5. **Concerning Score**: ...
6. **User Frustration**: ...
7. **Assistant Errors**: ...
```

---

## Output: `ConversationSummary`

The result of summarization is a `ConversationSummary` object (see `kura/types/summarisation.py`):

```python
class ConversationSummary(BaseModel):
    chat_id: str
    summary: str
    request: Optional[str]
    languages: Optional[list[str]]
    task: Optional[str]
    concerning_score: Optional[int]  # 1–5
    user_frustration: Optional[int]  # 1–5
    assistant_errors: Optional[list[str]]
    metadata: dict
    embedding: Optional[list[float]] = None
```

- **chat_id**: Unique conversation ID
- **summary**: Concise summary (max 2 sentences, no PII)
- **request**: User's overall request
- **languages**: List of languages (e.g., `['english', 'python']`)
- **task**: Main task
- **concerning_score**: Safety concern (1 = benign, 5 = urgent)
- **user_frustration**: User frustration (1 = happy, 5 = extremely annoyed)
- **assistant_errors**: List of assistant errors
- **metadata**: Additional metadata (e.g., conversation turns, custom extractors)
- **embedding**: Optional vector embedding for clustering/search

---

## Pipeline Integration

Summarization is the first major step in Kura's analysis pipeline:

1. **Loading**: Conversations are loaded from various sources
2. **Summarization**: Each conversation is summarized as above
3. **Embedding**: Summaries are embedded as vectors
4. **Clustering**: Similar summaries are grouped
5. **Visualization/Analysis**: Clusters and summaries are explored

---

## References

- [Clio: Privacy-Preserving Insights into Real-World AI Use (Anthropic)](https://assets.anthropic.com/m/7e1ab885d1b24176/original/Clio-Privacy-Preserving-Insights-into-Real-World-AI-Use.pdf)
- [API documentation](../api/index.md)
- [Conversations](conversations.md)
