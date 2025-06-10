#!/usr/bin/env python3
"""
Test script for MiniBatch K-means clustering implementation.

This script demonstrates the usage of the new MiniBatchKmeansClusteringMethod
and compares it with the standard KmeansClusteringMethod.

Run with: python test_minibatch_kmeans.py
"""

import numpy as np
from kura.k_means import KmeansClusteringMethod, MiniBatchKmeansClusteringMethod
import logging

# Set up logging to see the clustering progress
logging.basicConfig(level=logging.INFO)


def create_test_data(n_items: int = 1000) -> list[dict]:
    """Create synthetic test data with embeddings."""
    np.random.seed(42)

    # Create 3 distinct clusters in 2D space for visualization
    cluster1 = np.random.normal([0, 0], 0.5, (n_items // 3, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (n_items // 3, 2))
    cluster3 = np.random.normal([-2, 4], 0.5, (n_items - 2 * (n_items // 3), 2))

    embeddings = np.vstack([cluster1, cluster2, cluster3])

    # Create test items with embeddings
    items = []
    for i, embedding in enumerate(embeddings):
        items.append({"embedding": embedding.tolist(), "item": f"test_item_{i}"})

    return items


def test_clustering_methods():
    """Test both clustering methods and compare results."""
    print("=== MiniBatch K-means Clustering Test ===\n")

    # Create test data
    test_data = create_test_data(1000)
    print(f"Created {len(test_data)} test items with 2D embeddings")

    # Test Standard K-means
    print("\n--- Testing Standard K-means ---")
    standard_kmeans = KmeansClusteringMethod(clusters_per_group=50)
    standard_results = standard_kmeans.cluster(test_data)

    print(f"Standard K-means created {len(standard_results)} clusters")
    standard_sizes = [len(items) for items in standard_results.values()]
    print(
        f"Cluster sizes: min={min(standard_sizes)}, max={max(standard_sizes)}, avg={np.mean(standard_sizes):.1f}"
    )

    # Test MiniBatch K-means
    print("\n--- Testing MiniBatch K-means ---")
    minibatch_kmeans = MiniBatchKmeansClusteringMethod(
        clusters_per_group=50, batch_size=100, max_iter=50, random_state=42
    )
    minibatch_results = minibatch_kmeans.cluster(test_data)

    print(f"MiniBatch K-means created {len(minibatch_results)} clusters")
    minibatch_sizes = [len(items) for items in minibatch_results.values()]
    print(
        f"Cluster sizes: min={min(minibatch_sizes)}, max={max(minibatch_sizes)}, avg={np.mean(minibatch_sizes):.1f}"
    )

    # Compare results
    print("\n--- Comparison ---")
    print(f"Standard K-means clusters: {len(standard_results)}")
    print(f"MiniBatch K-means clusters: {len(minibatch_results)}")
    print(
        f"Difference in cluster count: {abs(len(standard_results) - len(minibatch_results))}"
    )

    # Test with larger dataset to show memory efficiency
    print("\n--- Testing with larger dataset (10k items) ---")
    large_test_data = create_test_data(10000)

    print("Testing MiniBatch K-means on large dataset...")
    large_minibatch_kmeans = MiniBatchKmeansClusteringMethod(
        clusters_per_group=100, batch_size=500, random_state=42
    )
    large_results = large_minibatch_kmeans.cluster(large_test_data)

    print(
        f"MiniBatch K-means processed {len(large_test_data)} items into {len(large_results)} clusters"
    )
    large_sizes = [len(items) for items in large_results.values()]
    print(
        f"Large dataset cluster sizes: min={min(large_sizes)}, max={max(large_sizes)}, avg={np.mean(large_sizes):.1f}"
    )


def test_edge_cases():
    """Test edge cases for MiniBatch K-means."""
    print("\n=== Edge Case Tests ===")

    minibatch_kmeans = MiniBatchKmeansClusteringMethod()

    # Test empty input
    print("Testing empty input...")
    empty_result = minibatch_kmeans.cluster([])
    print(f"Empty input result: {empty_result}")

    # Test single item
    print("Testing single item...")
    single_item = [{"embedding": [1.0, 2.0], "item": "single"}]
    single_result = minibatch_kmeans.cluster(single_item)
    print(f"Single item result: {len(single_result)} clusters")

    # Test small batch size
    print("Testing with very small batch size...")
    small_batch_kmeans = MiniBatchKmeansClusteringMethod(batch_size=1)
    small_data = create_test_data(10)
    small_result = small_batch_kmeans.cluster(small_data)
    print(f"Small batch result: {len(small_result)} clusters")


if __name__ == "__main__":
    try:
        test_clustering_methods()
        test_edge_cases()
        print("\n✅ All tests completed successfully!")
        print("\nThe MiniBatchKmeansClusteringMethod is ready for use!")
        print("Use it for large datasets (100k+ conversations) to avoid memory issues.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
