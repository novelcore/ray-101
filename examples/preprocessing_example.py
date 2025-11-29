"""
Example: Distributed Data Preprocessing with Ray Data

This example shows how to preprocess large datasets in parallel
using Ray Data on the cluster.
"""

import ray
import ray.data
import numpy as np
from typing import Dict, Any


def normalize_features(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize feature columns"""
    # Example: normalize to [0, 1]
    if "features" in batch:
        features = batch["features"]
        min_val = features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        batch["features"] = (features - min_val) / (max_val - min_val + 1e-8)
    return batch


def filter_valid(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out invalid rows"""
    # Example: keep rows where target is not NaN
    if "target" in batch:
        valid_mask = ~np.isnan(batch["target"])
        for key in batch:
            batch[key] = batch[key][valid_mask]
    return batch


def augment_data(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Data augmentation"""
    # Example: add noise
    if "features" in batch:
        noise = np.random.normal(0, 0.01, batch["features"].shape)
        batch["features_augmented"] = batch["features"] + noise
    return batch


def main():
    # Connect to Ray cluster
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True)
    
    print("✅ Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")
    
    # Create sample dataset
    # In practice, load from files:
    # dataset = ray.data.read_parquet("s3://bucket/data/*.parquet")
    # dataset = ray.data.read_csv("data/*.csv")
    
    print("\n📦 Creating sample dataset...")
    data = [
        {
            "features": np.random.randn(10).tolist(),
            "target": np.random.randn(1).item(),
            "id": i
        }
        for i in range(10000)
    ]
    
    dataset = ray.data.from_items(data)
    print(f"Initial dataset size: {dataset.count()}")
    
    # Preprocessing pipeline
    print("\n🔄 Running preprocessing pipeline...")
    
    processed = (
        dataset
        .map_batches(normalize_features, batch_size=1000)
        .map_batches(filter_valid, batch_size=1000)
        .map_batches(augment_data, batch_size=1000)
    )
    
    # Execute pipeline
    processed.fully_executed()
    
    print(f"Processed dataset size: {processed.count()}")
    
    # Show sample
    print("\n📊 Sample processed data:")
    sample = processed.take(3)
    for item in sample:
        print(f"  ID: {item['id']}, Features shape: {len(item.get('features', []))}")
    
    # Save processed data
    # processed.write_parquet("output/processed_data.parquet")
    # processed.write_csv("output/processed_data.csv")
    
    print("\n✅ Preprocessing complete!")
    
    ray.shutdown()


if __name__ == "__main__":
    main()

