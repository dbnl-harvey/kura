# Checkpoints

Kura's checkpoint system enables efficient caching and resumption of pipeline operations by saving intermediate results to disk. This prevents expensive recomputation when rerunning analyses and provides a foundation for incremental processing workflows.

---

## Overview

Checkpointing in Kura saves pipeline outputs at each major step:

1. **Conversation Summaries** - Generated from raw conversations
2. **Base Clusters** - Initial groupings of similar summaries
3. **Meta Clusters** - Hierarchical cluster reductions
4. **Projected Clusters** - 2D coordinates for visualization

The checkpoint system supports two storage formats: **JSONL** (default) and **Parquet** (optimized).

---

## JSONL Format (Default)

The default checkpoint format uses JSON Lines (JSONL), where each line contains a complete JSON object representing one data item.

### Characteristics

- **Human-readable** - Easy to inspect and debug
- **Text-based** - Works with standard text processing tools
- **Universal compatibility** - Supported everywhere JSON is supported
- **Streaming friendly** - Can process line-by-line without loading entire file
- **Simple implementation** - Minimal dependencies

### Example Structure

```jsonl
{"chat_id":"abc123","summary":"User seeks Python help","embedding":[0.1,0.2,0.3],"metadata":{"turns":3}}
{"chat_id":"def456","summary":"Data analysis question","embedding":[0.4,0.5,0.6],"metadata":{"turns":2}}
```

### Usage

```python
from kura import CheckpointManager

# Default JSONL checkpointing
checkpoint_manager = CheckpointManager("./checkpoints", enabled=True)

# Use in pipeline functions
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```

---

## Parquet Format (Optimized)

Parquet is a columnar storage format optimized for analytical workloads, offering significant compression and performance benefits.

### Characteristics

- **Columnar storage** - Optimized for analytical queries
- **Built-in compression** - Typically 50-80% smaller files
- **Schema evolution** - Self-describing with type information
- **Fast loading** - Optimized for data science workflows
- **Ecosystem compatibility** - Works with pandas, polars, Spark, etc.

### Compression Benefits

Real-world compression results from Kura pipeline:

| Data Type | JSONL Size | Parquet Size | Reduction | Ratio |
|-----------|------------|--------------|-----------|-------|
| Summaries | 126KB | 46KB | 63.5% | 2.7x |
| Dimensionality | 30KB | 14KB | 53.3% | 2.1x |
| Meta Clusters | 29KB | 13KB | 55.2% | 2.2x |
| Clusters | 16KB | 13KB | 18.8% | 1.2x |

### Usage

```python
from kura import ParquetCheckpointManager

# Parquet checkpointing (requires PyArrow)
checkpoint_manager = ParquetCheckpointManager("./checkpoints", enabled=True)

# Use in pipeline functions
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```

### Requirements

```bash
# Install PyArrow for Parquet support
pip install pyarrow
# or with uv
uv add pyarrow
```

---

## Format Comparison

| Aspect | JSONL | Parquet |
|--------|-------|---------|
| **File Size** | Larger | 50-80% smaller |
| **Loading Speed** | Good | Faster for large files |
| **Human Readable** | ✅ Yes | ❌ Binary format |
| **Compression** | Text compression only | Built-in columnar compression |
| **Dependencies** | None | Requires PyArrow |
| **Ecosystem** | Universal JSON support | Data science focused |
| **Streaming** | ✅ Line-by-line | ❌ Requires full load |
| **Schema** | Implicit | Explicit with types |

---

## When to Use Which Format

### Choose JSONL When:

- **Development and debugging** - Need to inspect checkpoint contents
- **Small datasets** - File size isn't a concern (< 100MB)
- **Minimal dependencies** - Want to avoid PyArrow requirement
- **Text processing workflows** - Need to use grep, awk, sed, etc.
- **Universal compatibility** - Sharing with non-Python systems

### Choose Parquet When:

- **Production workflows** - Optimizing for performance and storage
- **Large datasets** - Significant file size reduction needed
- **Data science integration** - Working with pandas, Spark, etc.
- **Analytical workloads** - Frequent filtering and aggregation
- **Cloud storage** - Minimizing transfer and storage costs

---

## Configuration Options

### JSONL Checkpoint Manager

```python
checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    enabled=True  # Set to False to disable checkpointing
)
```

### Parquet Checkpoint Manager

```python
checkpoint_manager = ParquetCheckpointManager(
    checkpoint_dir="./checkpoints",
    enabled=True,
    compression='snappy'  # Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
)
```

#### Compression Algorithms

- **snappy** (default) - Fast compression/decompression, good compression ratio
- **gzip** - Better compression, slower than snappy
- **brotli** - Excellent compression, slower processing
- **lz4** - Fastest compression, lower ratio
- **zstd** - Good balance of speed and compression

---

## File Organization

Both formats organize checkpoints in the same directory structure:

```
checkpoints/
├── summaries.jsonl           # or summaries.parquet
├── clusters.jsonl           # or clusters.parquet
├── meta_clusters.jsonl      # or meta_clusters.parquet
└── dimensionality.jsonl     # or dimensionality.parquet
```

---

## Performance Characteristics

### JSONL Performance

- **Write speed**: Fast for small files, slower for large datasets
- **Read speed**: Good, can stream process line-by-line
- **Memory usage**: Low during streaming, high if loading all at once
- **Compression**: Relies on external gzip compression

### Parquet Performance

- **Write speed**: Slower initial write due to compression
- **Read speed**: Very fast, especially for analytical queries
- **Memory usage**: Efficient columnar representation
- **Compression**: Built-in, optimized for data types

### Benchmark Example

For a dataset with 1000 conversation summaries:

| Operation | JSONL | Parquet | Improvement |
|-----------|-------|---------|-------------|
| File size | 2.4MB | 0.9MB | 62% smaller |
| Write time | 0.8s | 1.2s | 50% slower |
| Read time | 0.3s | 0.1s | 3x faster |
| Memory usage | 45MB | 28MB | 38% less |

---

## Migration Between Formats

You can easily convert between formats:

```python
from kura import CheckpointManager, ParquetCheckpointManager
from kura.types import ConversationSummary

# Load from JSONL
jsonl_manager = CheckpointManager("./jsonl_checkpoints")
summaries = jsonl_manager.load_checkpoint("summaries.jsonl", ConversationSummary)

# Save to Parquet
parquet_manager = ParquetCheckpointManager("./parquet_checkpoints")
parquet_manager.save_checkpoint("summaries.parquet", summaries)
```

---

## Best Practices

### Development Workflow

1. **Start with JSONL** for initial development and debugging
2. **Switch to Parquet** for production runs with large datasets
3. **Use JSONL** for sharing examples and test cases
4. **Use Parquet** for long-term storage and analytical workflows

### Storage Management

```python
# Monitor checkpoint sizes
import os

def get_checkpoint_sizes(directory):
    sizes = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            sizes[filename] = os.path.getsize(path)
    return sizes

# Compare formats
jsonl_sizes = get_checkpoint_sizes("./jsonl_checkpoints")
parquet_sizes = get_checkpoint_sizes("./parquet_checkpoints")
```

### Error Handling

Both formats handle errors gracefully:

```python
try:
    summaries = checkpoint_manager.load_checkpoint("summaries", ConversationSummary)
    if summaries is None:
        print("No checkpoint found, will generate from scratch")
except Exception as e:
    print(f"Checkpoint loading failed: {e}")
    # Fallback to regeneration
```

---

## References

- [Parquet Format Documentation](https://parquet.apache.org/docs/)
- [JSON Lines Specification](https://jsonlines.org/)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)
- [Kura Pipeline Overview](overview.md)
