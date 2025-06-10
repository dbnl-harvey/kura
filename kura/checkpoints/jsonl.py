"""
JSONL-based checkpoint manager.

This is the traditional checkpoint system that uses JSONL files for storing
intermediate pipeline results. This is kept for backward compatibility.
"""

import logging
from typing import Optional, TypeVar, List
import os
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class JSONLCheckpointManager:
    """Handles checkpoint loading and saving using JSONL files.
    
    This is the original checkpoint system that stores each checkpoint as
    a JSONL file where each line is a JSON-serialized Pydantic model.
    """

    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
        """
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled

        if self.enabled:
            self.setup_checkpoint_dir()

    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

    def get_checkpoint_path(self, filename: str) -> str:
        """Get full path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, filename)

    def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
        """Load data from a checkpoint file if it exists.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        checkpoint_path = self.get_checkpoint_path(filename)
        if os.path.exists(checkpoint_path):
            logger.info(
                f"Loading checkpoint from {checkpoint_path} for {model_class.__name__}"
            )
            with open(checkpoint_path, "r") as f:
                return [model_class.model_validate_json(line) for line in f]
        return None

    def save_checkpoint(self, filename: str, data: List[T]) -> None:
        """Save data to a checkpoint file.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
        """
        if not self.enabled:
            return

        checkpoint_path = self.get_checkpoint_path(filename)
        with open(checkpoint_path, "w") as f:
            for item in data:
                f.write(item.model_dump_json() + "\n")
        logger.info(f"Saved checkpoint to {checkpoint_path} with {len(data)} items")

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint files."""
        if not self.enabled or not os.path.exists(self.checkpoint_dir):
            return []
        
        return [
            f for f in os.listdir(self.checkpoint_dir) 
            if f.endswith(('.jsonl', '.json'))
        ]

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint file.
        
        Args:
            filename: Name of the checkpoint file to delete
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        if not self.enabled:
            return False
            
        checkpoint_path = self.get_checkpoint_path(filename)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        return False