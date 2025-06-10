#!/usr/bin/env python3
"""
Simple test script to verify ParquetCheckpointManager implementation.
"""

import tempfile
import asyncio
from pathlib import Path

# Test imports
try:
    from kura.v1.parquet_checkpoint import ParquetCheckpointManager
    from kura.types import Conversation, ConversationSummary, Cluster
    from kura.types.dimensionality import ProjectedCluster
    from datetime import datetime
    import uuid
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def create_sample_data():
    """Create sample data for testing."""
    # Sample conversation
    conversation = Conversation(
        id=str(uuid.uuid4()),
        chat_id="test_chat_1",
        created_at=datetime.now(),
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?",
                "created_at": str(datetime.now())
            },
            {
                "role": "assistant", 
                "content": "I'm doing well, thank you!",
                "created_at": str(datetime.now())
            }
        ],
        metadata={"test": "value"}
    )
    
    # Sample summary
    summary = ConversationSummary(
        chat_id="test_chat_1",
        summary="A friendly greeting conversation",
        embedding=[0.1, 0.2, 0.3] * 512,  # 1536 dimensional embedding
        extracted_properties=[]
    )
    
    # Sample cluster
    cluster = Cluster(
        id=str(uuid.uuid4()),
        summaries=["test_chat_1"],
        cluster_summary="Greeting conversations",
        embedding=[0.5, 0.6, 0.7] * 512,
        parent_id=None,
        children=[]
    )
    
    # Sample projected cluster
    projected_cluster = ProjectedCluster(
        id=cluster.id,
        summaries=cluster.summaries,
        cluster_summary=cluster.cluster_summary,
        embedding=cluster.embedding,
        parent_id=cluster.parent_id,
        children=cluster.children,
        x=1.5,
        y=2.5
    )
    
    return conversation, summary, cluster, projected_cluster

def test_parquet_checkpoint_manager():
    """Test ParquetCheckpointManager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint manager
        manager = ParquetCheckpointManager(temp_dir, enabled=True)
        print(f"✅ Created ParquetCheckpointManager in {temp_dir}")
        
        # Create sample data
        conversation, summary, cluster, projected_cluster = create_sample_data()
        
        # Test conversation checkpointing
        manager.save_checkpoint("conversations.jsonl", [conversation])
        loaded_conversations = manager.load_checkpoint("conversations.jsonl", Conversation)
        assert len(loaded_conversations) == 1
        assert loaded_conversations[0].chat_id == conversation.chat_id
        print("✅ Conversation checkpoint save/load successful")
        
        # Test summary checkpointing
        manager.save_checkpoint("summaries.jsonl", [summary])
        loaded_summaries = manager.load_checkpoint("summaries.jsonl", ConversationSummary)
        assert len(loaded_summaries) == 1
        assert loaded_summaries[0].chat_id == summary.chat_id
        assert len(loaded_summaries[0].embedding) == len(summary.embedding)
        print("✅ Summary checkpoint save/load successful")
        
        # Test cluster checkpointing
        manager.save_checkpoint("clusters.jsonl", [cluster])
        loaded_clusters = manager.load_checkpoint("clusters.jsonl", Cluster)
        assert len(loaded_clusters) == 1
        assert loaded_clusters[0].id == cluster.id
        print("✅ Cluster checkpoint save/load successful")
        
        # Test projected cluster checkpointing
        manager.save_checkpoint("projected_clusters.jsonl", [projected_cluster])
        loaded_projected = manager.load_checkpoint("projected_clusters.jsonl", ProjectedCluster)
        assert len(loaded_projected) == 1
        assert loaded_projected[0].x == projected_cluster.x
        assert loaded_projected[0].y == projected_cluster.y
        print("✅ Projected cluster checkpoint save/load successful")
        
        # Test file size reporting
        conv_size = manager.get_file_size("conversations.jsonl")
        summary_size = manager.get_file_size("summaries.jsonl")
        cluster_size = manager.get_file_size("clusters.jsonl")
        projected_size = manager.get_file_size("projected_clusters.jsonl")
        
        print(f"✅ File sizes: conversations={conv_size}B, summaries={summary_size}B, clusters={cluster_size}B, projected={projected_size}B")
        
        # Test checkpoint listing
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 4
        print(f"✅ Listed checkpoints: {checkpoints}")
        
        print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_parquet_checkpoint_manager()