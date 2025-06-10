"""
Migration utilities for converting between checkpoint formats.

This module provides tools to migrate from the legacy JSONL checkpoint
format to the new HuggingFace datasets format.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from .jsonl import JSONLCheckpointManager
from .hf_dataset import HFDatasetCheckpointManager
from kura.types import Conversation, ConversationSummary, Cluster
from kura.types.dimensionality import ProjectedCluster

logger = logging.getLogger(__name__)


def migrate_jsonl_to_hf_dataset(
    source_dir: str,
    target_dir: str,
    *,
    hub_repo: Optional[str] = None,
    hub_token: Optional[str] = None,
    compression: Optional[str] = "gzip",
    delete_source: bool = False
) -> Dict[str, bool]:
    """Migrate JSONL checkpoints to HuggingFace datasets format.
    
    Args:
        source_dir: Directory containing JSONL checkpoints
        target_dir: Directory for new HF dataset checkpoints
        hub_repo: Optional HuggingFace Hub repository for upload
        hub_token: Optional HuggingFace Hub token for authentication
        compression: Compression algorithm for datasets
        delete_source: Whether to delete source JSONL files after migration
        
    Returns:
        Dictionary mapping checkpoint names to migration success status
    """
    logger.info(f"Starting migration from {source_dir} to {target_dir}")
    
    # Initialize managers
    jsonl_manager = JSONLCheckpointManager(source_dir, enabled=True)
    hf_manager = HFDatasetCheckpointManager(
        target_dir,
        enabled=True,
        hub_repo=hub_repo,
        hub_token=hub_token,
        compression=compression
    )
    
    # Define checkpoint types and their corresponding model classes
    checkpoint_mappings = {
        "conversations.json": (Conversation, "conversations"),
        "summaries.jsonl": (ConversationSummary, "summaries"),
        "clusters.jsonl": (Cluster, "clusters"),
        "meta_clusters.jsonl": (Cluster, "clusters"),
        "dimensionality.jsonl": (ProjectedCluster, "projected_clusters"),
    }
    
    results = {}
    
    # Find all available checkpoints
    available_checkpoints = jsonl_manager.list_checkpoints()
    logger.info(f"Found {len(available_checkpoints)} JSONL checkpoints to migrate")
    
    for checkpoint_file in available_checkpoints:
        try:
            # Determine model class and checkpoint type
            model_class = None
            checkpoint_type = None
            checkpoint_name = None
            
            for file_pattern, (cls, ctype) in checkpoint_mappings.items():
                if checkpoint_file.endswith(file_pattern.split('.')[-1]) or checkpoint_file == file_pattern:
                    model_class = cls
                    checkpoint_type = ctype
                    # Use the pattern name without extension as checkpoint name
                    checkpoint_name = file_pattern.split('.')[0]
                    break
            
            if not model_class or not checkpoint_type:
                logger.warning(f"Unknown checkpoint type for {checkpoint_file}, skipping")
                results[checkpoint_file] = False
                continue
                
            # Handle special case for conversations.json (not JSONL format)
            if checkpoint_file == "conversations.json":
                # Load JSON format for conversations
                source_path = Path(source_dir) / checkpoint_file
                if not source_path.exists():
                    logger.warning(f"Checkpoint {checkpoint_file} not found, skipping")
                    results[checkpoint_file] = False
                    continue
                    
                with open(source_path, 'r') as f:
                    conversation_data = json.load(f)
                    
                conversations = [Conversation.model_validate(item) for item in conversation_data]
                logger.info(f"Loaded {len(conversations)} conversations from {checkpoint_file}")
                
            else:
                # Load JSONL format
                data = jsonl_manager.load_checkpoint(checkpoint_file, model_class)
                if not data:
                    logger.warning(f"Failed to load {checkpoint_file}, skipping")
                    results[checkpoint_file] = False
                    continue
                    
                conversations = data
                logger.info(f"Loaded {len(conversations)} items from {checkpoint_file}")
            
            # Save to HuggingFace dataset format
            hf_manager.save_checkpoint(checkpoint_name, conversations, checkpoint_type)
            
            # Verify the migration
            loaded_data = hf_manager.load_checkpoint(checkpoint_name, model_class, checkpoint_type=checkpoint_type)
            if loaded_data and len(loaded_data) == len(conversations):
                logger.info(f"Successfully migrated {checkpoint_file} -> {checkpoint_name}")
                results[checkpoint_file] = True
                
                # Delete source file if requested
                if delete_source:
                    if jsonl_manager.delete_checkpoint(checkpoint_file):
                        logger.info(f"Deleted source file: {checkpoint_file}")
            else:
                logger.error(f"Migration verification failed for {checkpoint_file}")
                results[checkpoint_file] = False
                
        except Exception as e:
            logger.error(f"Failed to migrate {checkpoint_file}: {e}")
            results[checkpoint_file] = False
    
    successful = sum(results.values())
    total = len(results)
    logger.info(f"Migration complete: {successful}/{total} checkpoints migrated successfully")
    
    return results


def verify_migration(
    jsonl_dir: str,
    hf_dir: str,
    *,
    detailed: bool = False
) -> Dict[str, Any]:
    """Verify that a migration was successful by comparing data.
    
    Args:
        jsonl_dir: Directory containing original JSONL checkpoints
        hf_dir: Directory containing migrated HF dataset checkpoints
        detailed: Whether to perform detailed comparison of data content
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"Verifying migration from {jsonl_dir} to {hf_dir}")
    
    jsonl_manager = JSONLCheckpointManager(jsonl_dir, enabled=True)
    hf_manager = HFDatasetCheckpointManager(hf_dir, enabled=True)
    
    checkpoint_mappings = {
        "conversations.json": (Conversation, "conversations"),
        "summaries.jsonl": (ConversationSummary, "summaries"),
        "clusters.jsonl": (Cluster, "clusters"),
        "meta_clusters.jsonl": (Cluster, "clusters"),
        "dimensionality.jsonl": (ProjectedCluster, "projected_clusters"),
    }
    
    results = {
        "total_checkpoints": 0,
        "verified_checkpoints": 0,
        "failed_checkpoints": [],
        "details": {}
    }
    
    available_checkpoints = jsonl_manager.list_checkpoints()
    results["total_checkpoints"] = len(available_checkpoints)
    
    for checkpoint_file in available_checkpoints:
        try:
            # Find corresponding mapping
            model_class = None
            checkpoint_type = None
            checkpoint_name = None
            
            for file_pattern, (cls, ctype) in checkpoint_mappings.items():
                if checkpoint_file.endswith(file_pattern.split('.')[-1]) or checkpoint_file == file_pattern:
                    model_class = cls
                    checkpoint_type = ctype
                    checkpoint_name = file_pattern.split('.')[0]
                    break
            
            if not model_class:
                results["failed_checkpoints"].append(f"{checkpoint_file}: Unknown type")
                continue
            
            # Load from both sources
            if checkpoint_file == "conversations.json":
                source_path = Path(jsonl_dir) / checkpoint_file
                with open(source_path, 'r') as f:
                    conversation_data = json.load(f)
                jsonl_data = [Conversation.model_validate(item) for item in conversation_data]
            else:
                jsonl_data = jsonl_manager.load_checkpoint(checkpoint_file, model_class)
                
            hf_data = hf_manager.load_checkpoint(checkpoint_name, model_class, checkpoint_type=checkpoint_type)
            
            if not jsonl_data:
                results["failed_checkpoints"].append(f"{checkpoint_file}: JSONL data not found")
                continue
                
            if not hf_data:
                results["failed_checkpoints"].append(f"{checkpoint_file}: HF data not found")
                continue
            
            # Basic count verification
            if len(jsonl_data) != len(hf_data):
                results["failed_checkpoints"].append(
                    f"{checkpoint_file}: Count mismatch (JSONL: {len(jsonl_data)}, HF: {len(hf_data)})"
                )
                continue
            
            verification_result = {
                "count": len(jsonl_data),
                "count_match": True
            }
            
            # Detailed verification if requested
            if detailed:
                # Compare first few items for structure
                sample_size = min(5, len(jsonl_data))
                structural_matches = 0
                
                for i in range(sample_size):
                    jsonl_dict = jsonl_data[i].model_dump()
                    hf_dict = hf_data[i].model_dump()
                    
                    # Compare key fields (skip computed fields and timestamps that might have precision differences)
                    key_fields = []
                    if hasattr(jsonl_data[i], 'chat_id'):
                        key_fields.append('chat_id')
                    if hasattr(jsonl_data[i], 'id'):
                        key_fields.append('id')
                    if hasattr(jsonl_data[i], 'name'):
                        key_fields.append('name')
                    
                    matches = all(
                        jsonl_dict.get(field) == hf_dict.get(field)
                        for field in key_fields
                    )
                    
                    if matches:
                        structural_matches += 1
                
                verification_result["structural_match_rate"] = structural_matches / sample_size
                verification_result["detailed_verified"] = structural_matches == sample_size
            
            results["details"][checkpoint_file] = verification_result
            
            if not detailed or verification_result.get("detailed_verified", True):
                results["verified_checkpoints"] += 1
                logger.info(f"Verified {checkpoint_file} successfully")
            else:
                results["failed_checkpoints"].append(f"{checkpoint_file}: Detailed verification failed")
                
        except Exception as e:
            results["failed_checkpoints"].append(f"{checkpoint_file}: {str(e)}")
            logger.error(f"Failed to verify {checkpoint_file}: {e}")
    
    success_rate = results["verified_checkpoints"] / results["total_checkpoints"] if results["total_checkpoints"] > 0 else 0
    logger.info(f"Verification complete: {results['verified_checkpoints']}/{results['total_checkpoints']} checkpoints verified ({success_rate:.1%})")
    
    return results


def estimate_migration_benefits(jsonl_dir: str) -> Dict[str, Any]:
    """Estimate the benefits of migrating to HuggingFace datasets format.
    
    Args:
        jsonl_dir: Directory containing JSONL checkpoints
        
    Returns:
        Dictionary with estimated benefits and current usage stats
    """
    jsonl_manager = JSONLCheckpointManager(jsonl_dir, enabled=True)
    
    stats = {
        "current_format": "JSONL",
        "total_files": 0,
        "total_size_bytes": 0,
        "estimated_benefits": {
            "memory_efficiency": "10-100x better for large datasets",
            "loading_speed": "Significantly faster with memory mapping",
            "compression": "Built-in compression reduces storage by 50-80%",
            "streaming": "Process datasets larger than RAM",
            "querying": "Filter without loading entire dataset",
            "versioning": "Built-in version control via HuggingFace Hub",
            "collaboration": "Easy sharing via HuggingFace Hub"
        }
    }
    
    try:
        jsonl_dir_path = Path(jsonl_dir)
        if jsonl_dir_path.exists():
            for file_path in jsonl_dir_path.rglob("*.jsonl"):
                if file_path.is_file():
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += file_path.stat().st_size
            
            # Check for conversations.json
            conv_file = jsonl_dir_path / "conversations.json"
            if conv_file.exists():
                stats["total_files"] += 1
                stats["total_size_bytes"] += conv_file.stat().st_size
        
        # Provide size estimates
        size_mb = stats["total_size_bytes"] / (1024 * 1024)
        stats["total_size_mb"] = round(size_mb, 2)
        
        if size_mb > 100:
            stats["migration_priority"] = "HIGH - Large datasets will benefit significantly"
        elif size_mb > 10:
            stats["migration_priority"] = "MEDIUM - Moderate benefits expected"
        else:
            stats["migration_priority"] = "LOW - Small datasets, benefits mainly for future scalability"
            
        # Estimate compressed size
        stats["estimated_compressed_size_mb"] = round(size_mb * 0.3, 2)  # Assume 70% compression
        stats["estimated_space_savings_mb"] = round(size_mb * 0.7, 2)
        
    except Exception as e:
        logger.error(f"Failed to analyze directory {jsonl_dir}: {e}")
        stats["error"] = str(e)
    
    return stats