"""
Simple Ray Pipeline: Text -> Image -> 3D Object
===============================================

A lightweight pipeline that:
1. Takes text input
2. Generates image using diffusion model
3. Converts image to 3D object
"""

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Any


def text_to_image_worker(config: Dict[str, Any]):
    """Generate image from text using a simple diffusion-like model"""
    from ray.train import report
    
    text_prompt = config.get("text_prompt", "a red cube")
    
    # Simulate image generation (in production, use Stable Diffusion or similar)
    # For demo: create a simple colored image based on text
    print(f"🎨 Generating image from text: '{text_prompt}'")
    
    # Simple simulation - create a 64x64 RGB image
    # In real implementation, this would call a diffusion model
    image = np.random.rand(64, 64, 3) * 255
    image = image.astype(np.uint8)
    
    # Simulate processing time
    import time
    time.sleep(0.5)
    
    print(f"✅ Image generated: {image.shape}")
    
    report({
        "step": "text_to_image",
        "image_shape": image.shape,
        "prompt": text_prompt
    })
    
    return image


def image_to_3d_worker(config: Dict[str, Any]):
    """Convert image to 3D object representation"""
    from ray.train import report
    
    # In production, use models like Zero-1-to-3, Shap-E, or similar
    # For demo: create a simple 3D mesh representation
    print("🎲 Converting image to 3D object...")
    
    # Simulate 3D object generation
    # Create a simple point cloud or mesh
    vertices = np.random.rand(100, 3) * 10  # 100 vertices
    faces = np.random.randint(0, 100, (50, 3))  # 50 triangular faces
    
    # Simulate processing time
    import time
    time.sleep(0.3)
    
    print(f"✅ 3D object generated: {len(vertices)} vertices, {len(faces)} faces")
    
    report({
        "step": "image_to_3d",
        "vertices": len(vertices),
        "faces": len(faces)
    })
    
    return {"vertices": vertices, "faces": faces}


def simple_pipeline_worker(config: Dict[str, Any]):
    """Complete pipeline: text -> image -> 3D"""
    from ray.train import report
    
    text_prompt = config.get("text_prompt", "a red cube")
    
    print(f"\n{'='*60}")
    print(f"Pipeline: '{text_prompt}'")
    print(f"{'='*60}\n")
    
    # Step 1: Text to Image
    image = text_to_image_worker({"text_prompt": text_prompt})
    
    # Step 2: Image to 3D
    obj_3d = image_to_3d_worker({"image": image})
    
    # Final result
    result = {
        "prompt": text_prompt,
        "image_shape": image.shape,
        "3d_vertices": len(obj_3d["vertices"]),
        "3d_faces": len(obj_3d["faces"])
    }
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Text: '{text_prompt}'")
    print(f"   Image: {image.shape}")
    print(f"   3D Object: {len(obj_3d['vertices'])} vertices, {len(obj_3d['faces'])} faces")
    
    report(result)
    return result


def main():
    """Main pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Ray Pipeline: Text -> Image -> 3D")
    parser.add_argument("--text", type=str, default="a red cube", help="Text prompt for image generation")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of Ray workers")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Simple Ray Pipeline: Text -> Image -> 3D Object")
    print("=" * 60)
    
    # Initialize Ray - connect to cluster or run locally
    print("\n1. Initializing Ray...")
    cluster_address = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
    
    connected_to_cluster = False
    if cluster_address.startswith("ray://"):
        try:
            ray.init(address=cluster_address, ignore_reinit_error=True)
            print(f"   ✅ Connected to Ray cluster!")
            print(f"   Resources: {ray.available_resources()}")
            connected_to_cluster = True
        except Exception as e:
            print(f"   ⚠️  Could not connect to cluster ({type(e).__name__})")
            print(f"   Falling back to local mode...")
    
    if not connected_to_cluster:
        ray.init(ignore_reinit_error=True)
        print(f"   ✅ Running in local mode")
        print(f"   Resources: {ray.available_resources()}")
    
    # Configuration
    config = {
        "text_prompt": args.text,
    }
    
    # Use minimal resources to avoid memory issues
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=False,  # Disable GPU for simplicity
        trainer_resources={"CPU": 1},  # Limit CPU per worker
    )
    
    run_config = RunConfig(
        storage_path=os.path.abspath("./ray_results"),
        name="simple_pipeline",
    )
    
    # Create trainer
    print(f"\n2. Setting up pipeline for: '{args.text}'...")
    trainer = TorchTrainer(
        train_loop_per_worker=simple_pipeline_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    # Run pipeline
    print("\n3. Running pipeline...")
    print("   Monitor at: http://localhost:8266")
    print("-" * 60)
    
    result = trainer.fit()
    
    # Print results
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {result.path}")
    
    ray.shutdown()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

