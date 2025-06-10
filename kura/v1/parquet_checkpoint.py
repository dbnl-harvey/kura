"""
Parquet-based checkpoint manager for efficient storage of Kura pipeline data.

This module provides a ParquetCheckpointManager that stores checkpoint data
in Parquet format for better compression and faster loading compared to JSONL.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, TypeVar, Dict, Any, Union
import json

from pydantic import BaseModel

# PyArrow imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from kura.types import Conversation, Cluster, ConversationSummary
from kura.types.dimensionality import ProjectedCluster

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ParquetCheckpointManager:
    """Handles checkpoint loading and saving using Parquet format for efficient storage."""
    
    def __init__(self, checkpoint_dir: str, *, enabled: bool = True, compression: str = 'snappy'):
        """Initialize Parquet checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
            compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')
        """
        if not PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for ParquetCheckpointManager. "
                "Install with: pip install pyarrow"
            )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.compression = compression
        self.schemas = self._define_schemas()
        
        if self.enabled:
            self.setup_checkpoint_dir()
    
    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")
    
    def _define_schemas(self) -> Dict[str, pa.Schema]:
        """Define PyArrow schemas for different data types."""
        return {
            'conversations': pa.schema([
                ('id', pa.string()),
                ('chat_id', pa.string()),
                ('created_at', pa.string()),
                ('messages', pa.string()),  # JSON string
                ('metadata', pa.string()),  # JSON string
            ]),
            'summaries': pa.schema([
                ('chat_id', pa.string()),
                ('summary', pa.string()),
                ('embedding', pa.list_(pa.float32())),
                ('extracted_properties', pa.string()),  # JSON string
            ]),
            'clusters': pa.schema([
                ('id', pa.string()),
                ('summaries', pa.string()),  # JSON string of list
                ('cluster_summary', pa.string()),
                ('embedding', pa.list_(pa.float32())),
                ('parent_id', pa.string()),
                ('children', pa.string()),  # JSON string of list
            ]),
            'projected_clusters': pa.schema([
                ('id', pa.string()),
                ('summaries', pa.string()),  # JSON string of list
                ('cluster_summary', pa.string()),
                ('embedding', pa.list_(pa.float32())),
                ('parent_id', pa.string()),
                ('children', pa.string()),  # JSON string of list
                ('x', pa.float32()),
                ('y', pa.float32()),
            ])
        }
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """Get full path for a checkpoint file, converting to .parquet extension."""
        # Convert .jsonl extensions to .parquet
        if filename.endswith('.jsonl'):
            filename = filename[:-6] + '.parquet'
        elif not filename.endswith('.parquet'):
            filename = filename + '.parquet'
        
        return self.checkpoint_dir / filename
    
    def _serialize_for_parquet(self, data: List[T], data_type: str) -> Dict[str, List[Any]]:
        """Convert Pydantic models to Parquet-compatible format."""
        if not data:
            return {}
        
        if data_type == 'conversations':
            return self._serialize_conversations(data)
        elif data_type == 'summaries':
            return self._serialize_summaries(data)
        elif data_type in ['clusters', 'projected_clusters']:
            return self._serialize_clusters(data, data_type)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _serialize_conversations(self, conversations: List[Conversation]) -> Dict[str, List[Any]]:
        """Serialize conversations for Parquet storage."""
        return {
            'id': [c.id for c in conversations],
            'chat_id': [c.chat_id for c in conversations],
            'created_at': [str(c.created_at) for c in conversations],
            'messages': [json.dumps([msg.model_dump() for msg in c.messages]) for c in conversations],
            'metadata': [json.dumps(c.metadata or {}) for c in conversations],
        }
    
    def _serialize_summaries(self, summaries: List[ConversationSummary]) -> Dict[str, List[Any]]:
        """Serialize conversation summaries for Parquet storage."""
        return {
            'chat_id': [s.chat_id for s in summaries],
            'summary': [s.summary for s in summaries],
            'embedding': [s.embedding for s in summaries],
            'extracted_properties': [json.dumps([prop.model_dump() for prop in s.extracted_properties]) for s in summaries],
        }
    
    def _serialize_clusters(self, clusters: List[Union[Cluster, ProjectedCluster]], data_type: str) -> Dict[str, List[Any]]:
        """Serialize clusters for Parquet storage."""
        result = {
            'id': [c.id for c in clusters],
            'summaries': [json.dumps(c.summaries) for c in clusters],
            'cluster_summary': [c.cluster_summary for c in clusters],
            'embedding': [c.embedding for c in clusters],
            'parent_id': [c.parent_id for c in clusters],
            'children': [json.dumps(c.children) for c in clusters],
        }
        
        # Add x, y coordinates for projected clusters
        if data_type == 'projected_clusters':
            result['x'] = [c.x for c in clusters]
            result['y'] = [c.y for c in clusters]
        
        return result
    
    def _deserialize_from_parquet(self, table: pa.Table, model_class: type[T]) -> List[T]:
        """Convert Parquet table back to Pydantic models."""
        # Convert to pandas for easier handling
        df = table.to_pandas()
        
        if model_class == Conversation:
            return self._deserialize_conversations(df)
        elif model_class == ConversationSummary:
            return self._deserialize_summaries(df)
        elif model_class in [Cluster, ProjectedCluster]:
            return self._deserialize_clusters(df, model_class)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    
    def _deserialize_conversations(self, df) -> List[Conversation]:
        """Deserialize conversations from Parquet data."""
        conversations = []
        for _, row in df.iterrows():
            messages_data = json.loads(row['messages'])
            metadata = json.loads(row['metadata'])
            
            # Reconstruct conversation
            conversation_dict = {
                'id': row['id'],
                'chat_id': row['chat_id'],
                'created_at': row['created_at'],
                'messages': messages_data,
                'metadata': metadata,
            }
            conversations.append(Conversation(**conversation_dict))
        
        return conversations
    
    def _deserialize_summaries(self, df) -> List[ConversationSummary]:
        """Deserialize conversation summaries from Parquet data."""
        summaries = []
        for _, row in df.iterrows():
            extracted_props_data = json.loads(row['extracted_properties'])
            
            # Reconstruct summary
            summary_dict = {
                'chat_id': row['chat_id'],
                'summary': row['summary'],
                'embedding': row['embedding'],
                'extracted_properties': extracted_props_data,
            }
            summaries.append(ConversationSummary(**summary_dict))
        
        return summaries
    
    def _deserialize_clusters(self, df, model_class: type[T]) -> List[T]:
        """Deserialize clusters from Parquet data."""
        clusters = []
        for _, row in df.iterrows():
            summaries_data = json.loads(row['summaries'])
            children_data = json.loads(row['children'])
            
            # Reconstruct cluster
            cluster_dict = {
                'id': row['id'],
                'summaries': summaries_data,
                'cluster_summary': row['cluster_summary'],
                'embedding': row['embedding'],
                'parent_id': row['parent_id'],
                'children': children_data,
            }
            
            # Add x, y coordinates for projected clusters
            if model_class == ProjectedCluster:
                cluster_dict['x'] = row['x']
                cluster_dict['y'] = row['y']
            
            clusters.append(model_class(**cluster_dict))
        
        return clusters
    
    def _get_data_type(self, model_class: type[T]) -> str:
        """Determine data type from model class."""
        if model_class == Conversation:
            return 'conversations'
        elif model_class == ConversationSummary:
            return 'summaries'
        elif model_class == Cluster:
            return 'clusters'
        elif model_class == ProjectedCluster:
            return 'projected_clusters'
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    
    def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
        """Load data from a Parquet checkpoint file.
        
        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data
            
        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None
        
        checkpoint_path = self.get_checkpoint_path(filename)
        if not checkpoint_path.exists():
            return None
        
        logger.info(f"Loading checkpoint from {checkpoint_path} for {model_class.__name__}")
        
        try:
            # Read Parquet file
            table = pq.read_table(checkpoint_path)
            
            # Deserialize back to Pydantic models
            data = self._deserialize_from_parquet(table, model_class)
            
            logger.info(f"Loaded {len(data)} items from checkpoint")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def save_checkpoint(self, filename: str, data: List[T]) -> None:
        """Save data to a Parquet checkpoint file.
        
        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
        """
        if not self.enabled or not data:
            return
        
        checkpoint_path = self.get_checkpoint_path(filename)
        
        try:
            # Determine data type and get schema
            data_type = self._get_data_type(type(data[0]))
            schema = self.schemas[data_type]
            
            # Serialize data for Parquet
            serialized_data = self._serialize_for_parquet(data, data_type)
            
            # Create PyArrow table
            table = pa.table(serialized_data, schema=schema)
            
            # Write to Parquet file
            pq.write_table(
                table,
                checkpoint_path,
                compression=self.compression,
                write_statistics=True,
                use_dictionary=True,  # Use dictionary encoding for string columns
            )
            
            logger.info(f"Saved checkpoint to {checkpoint_path} with {len(data)} items")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            raise
    
    def get_file_size(self, filename: str) -> int:
        """Get the size of a checkpoint file in bytes.
        
        Args:
            filename: Name of the checkpoint file
            
        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        checkpoint_path = self.get_checkpoint_path(filename)
        if checkpoint_path.exists():
            return checkpoint_path.stat().st_size
        return 0
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint files.
        
        Returns:
            List of checkpoint filenames
        """
        if not self.checkpoint_dir.exists():
            return []
        
        return [f.name for f in self.checkpoint_dir.glob("*.parquet")]