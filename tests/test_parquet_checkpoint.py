import pytest
import tempfile
import os
from pathlib import Path
from typing import List
from datetime import datetime

try:
    import pyarrow as pa
    from kura.v1.parquet_checkpoint import ParquetCheckpointManager
    from kura.types import Conversation, ConversationSummary, Cluster, Message
    from kura.types.dimensionality import ProjectedCluster

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Skip all tests if PyArrow is not available
pytestmark = pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")


class TestParquetCheckpointManagerInitialization:
    """Test ParquetCheckpointManager initialization and setup."""

    def test_init_with_pyarrow_available(self):
        """Test initialization when PyArrow is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir, enabled=True)
            assert manager.checkpoint_dir == Path(temp_dir)
            assert manager.enabled
            assert manager.compression == "snappy"
            assert temp_dir in str(manager.checkpoint_dir)

    def test_init_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "new_checkpoints")
            ParquetCheckpointManager(checkpoint_path, enabled=True)
            assert os.path.exists(checkpoint_path)

    def test_init_disabled(self):
        """Test initialization with checkpointing disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir, enabled=False)
            assert manager.enabled is False

    def test_init_custom_compression(self):
        """Test initialization with custom compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir, compression="gzip")
            assert manager.compression == "gzip"


class TestParquetCheckpointManagerSchemas:
    """Test schema definitions."""

    def test_schemas_defined(self):
        """Test that all required schemas are defined."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            required_schemas = [
                "conversations",
                "summaries",
                "clusters",
                "projected_clusters",
            ]
            for schema_name in required_schemas:
                assert schema_name in manager.schemas
                assert isinstance(manager.schemas[schema_name], pa.Schema)


class TestParquetCheckpointManagerConversations:
    """Test conversation serialization and deserialization."""

    @pytest.fixture
    def sample_conversations(self) -> List[Conversation]:
        """Create sample conversations for testing."""
        messages = [
            Message(
                role="user",
                content="Hello, can you help me with Python?",
                created_at=datetime.now(),
            ),
            Message(
                role="assistant",
                content="Of course! I'd be happy to help you with Python.",
                created_at=datetime.now(),
            ),
        ]

        return [
            Conversation(
                chat_id="chat_1",
                created_at=datetime.now(),
                messages=messages,
                metadata={"source": "test"},
            ),
            Conversation(
                chat_id="chat_2",
                created_at=datetime.now(),
                messages=messages,
                metadata={"source": "test", "language": "python"},
            ),
        ]

    def test_save_and_load_conversations(self, sample_conversations):
        """Test saving and loading conversations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Save conversations
            manager.save_checkpoint("conversations.parquet", sample_conversations)

            # Load conversations
            loaded = manager.load_checkpoint("conversations.parquet", Conversation)

            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0].chat_id == "chat_1"
            assert loaded[1].chat_id == "chat_2"
            assert loaded[0].metadata["source"] == "test"
            assert loaded[1].metadata["language"] == "python"


class TestParquetCheckpointManagerSummaries:
    """Test conversation summary serialization and deserialization."""

    @pytest.fixture
    def sample_summaries(self) -> List[ConversationSummary]:
        """Create sample conversation summaries for testing."""
        return [
            ConversationSummary(
                chat_id="chat_1",
                summary="User asks for Python help",
                request="Help with Python programming",
                topic="programming",
                languages=["english", "python"],
                task="Provide Python assistance",
                concerning_score=1,
                user_frustration=1,
                assistant_errors=None,
                metadata={"turns": 2},
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            ConversationSummary(
                chat_id="chat_2",
                summary="Advanced Python discussion",
                request=None,
                topic="advanced programming",
                languages=["english", "python"],
                task=None,
                concerning_score=None,
                user_frustration=None,
                assistant_errors=["Too technical"],
                metadata={"turns": 5, "complexity": "high"},
                embedding=[0.6, 0.7, 0.8, 0.9, 1.0],
            ),
        ]

    def test_save_and_load_summaries(self, sample_summaries):
        """Test saving and loading conversation summaries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Save summaries
            manager.save_checkpoint("summaries.parquet", sample_summaries)

            # Load summaries
            loaded = manager.load_checkpoint("summaries.parquet", ConversationSummary)

            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0].chat_id == "chat_1"
            assert loaded[0].summary == "User asks for Python help"
            assert loaded[0].languages == ["english", "python"]
            assert loaded[0].concerning_score == 1
            assert loaded[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

            assert loaded[1].chat_id == "chat_2"
            assert loaded[1].request is None
            assert loaded[1].assistant_errors == ["Too technical"]
            assert loaded[1].metadata["complexity"] == "high"


class TestParquetCheckpointManagerClusters:
    """Test cluster serialization and deserialization."""

    @pytest.fixture
    def sample_clusters(self) -> List[Cluster]:
        """Create sample clusters for testing."""
        return [
            Cluster(
                id="cluster_1",
                name="Python Programming",
                description="Cluster about Python programming questions",
                slug="python_programming_help",
                chat_ids=["chat_1", "chat_2"],
                parent_id=None,
            ),
            Cluster(
                id="cluster_2",
                name="Web Development",
                description="Cluster about web development topics",
                slug="web_development_topics",
                chat_ids=["chat_3", "chat_4", "chat_5"],
                parent_id="cluster_1",
            ),
        ]

    def test_save_and_load_clusters(self, sample_clusters):
        """Test saving and loading clusters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Save clusters
            manager.save_checkpoint("clusters.parquet", sample_clusters)

            # Load clusters
            loaded = manager.load_checkpoint("clusters.parquet", Cluster)

            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0].id == "cluster_1"
            assert loaded[0].name == "Python Programming"
            assert loaded[0].chat_ids == ["chat_1", "chat_2"]
            assert loaded[0].parent_id is None

            assert loaded[1].parent_id == "cluster_1"
            assert len(loaded[1].chat_ids) == 3


class TestParquetCheckpointManagerProjectedClusters:
    """Test projected cluster serialization and deserialization."""

    @pytest.fixture
    def sample_projected_clusters(self) -> List[ProjectedCluster]:
        """Create sample projected clusters for testing."""
        return [
            ProjectedCluster(
                id="proj_1",
                name="Python Cluster",
                description="Python programming cluster",
                slug="python_cluster",
                chat_ids=["chat_1", "chat_2"],
                parent_id=None,
                x_coord=1.5,
                y_coord=2.3,
                level=0,
            ),
            ProjectedCluster(
                id="proj_2",
                name="Web Dev Cluster",
                description="Web development cluster",
                slug="web_dev_cluster",
                chat_ids=["chat_3"],
                parent_id="proj_1",
                x_coord=-0.8,
                y_coord=1.2,
                level=1,
            ),
        ]

    def test_save_and_load_projected_clusters(self, sample_projected_clusters):
        """Test saving and loading projected clusters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Save projected clusters
            manager.save_checkpoint("projected.parquet", sample_projected_clusters)

            # Load projected clusters
            loaded = manager.load_checkpoint("projected.parquet", ProjectedCluster)

            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0].x_coord == 1.5
            assert loaded[0].y_coord == 2.3
            assert loaded[0].level == 0

            assert loaded[1].x_coord == -0.8
            assert loaded[1].level == 1


class TestParquetCheckpointManagerFileOperations:
    """Test file operations and utilities."""

    def test_get_checkpoint_path(self):
        """Test checkpoint path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Test .jsonl extension conversion
            path = manager.get_checkpoint_path("test.jsonl")
            assert str(path).endswith("test.parquet")

            # Test .parquet extension preservation
            path = manager.get_checkpoint_path("test.parquet")
            assert str(path).endswith("test.parquet")

            # Test extension addition
            path = manager.get_checkpoint_path("test")
            assert str(path).endswith("test.parquet")

    def test_get_file_size(self):
        """Test file size reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Non-existent file should return 0
            size = manager.get_file_size("nonexistent.parquet")
            assert size == 0

            # Create a test file and check size
            sample_data = [
                ConversationSummary(
                    chat_id="test",
                    summary="Test summary",
                    metadata={},
                    embedding=[1.0, 2.0, 3.0],
                )
            ]
            manager.save_checkpoint("test.parquet", sample_data)
            size = manager.get_file_size("test.parquet")
            assert size > 0

    def test_list_checkpoints(self):
        """Test listing checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Empty directory
            checkpoints = manager.list_checkpoints()
            assert checkpoints == []

            # Create some checkpoint files
            sample_data = [
                ConversationSummary(
                    chat_id="test", summary="Test", metadata={}, embedding=[1.0]
                )
            ]

            manager.save_checkpoint("test1.parquet", sample_data)
            manager.save_checkpoint("test2.parquet", sample_data)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            assert "test1.parquet" in checkpoints
            assert "test2.parquet" in checkpoints


class TestParquetCheckpointManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_disabled_checkpoint_manager(self):
        """Test that disabled manager doesn't save/load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir, enabled=False)

            sample_data = [
                ConversationSummary(
                    chat_id="test", summary="Test", metadata={}, embedding=[1.0]
                )
            ]

            # Save should do nothing when disabled
            manager.save_checkpoint("test.parquet", sample_data)

            # Load should return None when disabled
            loaded = manager.load_checkpoint("test.parquet", ConversationSummary)
            assert loaded is None

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            loaded = manager.load_checkpoint("nonexistent.parquet", ConversationSummary)
            assert loaded is None

    def test_save_empty_data(self):
        """Test saving empty data list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Should not create file when data is empty
            manager.save_checkpoint("empty.parquet", [])

            loaded = manager.load_checkpoint("empty.parquet", ConversationSummary)
            assert loaded is None

    def test_unknown_data_type(self):
        """Test handling of unknown data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            class UnknownModel:
                pass

            with pytest.raises(ValueError, match="Unknown model class"):
                manager._get_data_type(UnknownModel)


class TestParquetCheckpointManagerIntegration:
    """Integration tests comparing JSONL and Parquet formats."""

    def test_data_integrity_roundtrip(self):
        """Test that data survives serialization/deserialization intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Create comprehensive test data
            original_summaries = [
                ConversationSummary(
                    chat_id="comprehensive_test",
                    summary="This is a comprehensive test with all fields populated",
                    request="Test all functionality",
                    topic="testing",
                    languages=["english", "python", "javascript"],
                    task="Validate data integrity",
                    concerning_score=2,
                    user_frustration=3,
                    assistant_errors=["Minor error 1", "Minor error 2"],
                    metadata={
                        "complex_nested": {"level1": {"level2": ["item1", "item2"]}},
                        "numeric_value": 42,
                        "boolean_flag": True,
                        "null_value": None,
                    },
                    embedding=[i * 0.1 for i in range(100)],  # Large embedding
                )
            ]

            # Save and reload
            manager.save_checkpoint("integrity_test.parquet", original_summaries)
            loaded_summaries = manager.load_checkpoint(
                "integrity_test.parquet", ConversationSummary
            )

            assert loaded_summaries is not None
            assert len(loaded_summaries) == 1

            original = original_summaries[0]
            loaded = loaded_summaries[0]

            # Verify all fields match exactly
            assert loaded.chat_id == original.chat_id
            assert loaded.summary == original.summary
            assert loaded.request == original.request
            assert loaded.topic == original.topic
            assert loaded.languages == original.languages
            assert loaded.task == original.task
            assert loaded.concerning_score == original.concerning_score
            assert loaded.user_frustration == original.user_frustration
            assert loaded.assistant_errors == original.assistant_errors
            assert loaded.metadata == original.metadata
            assert loaded.embedding == original.embedding
