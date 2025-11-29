"""
Ray ML Pipeline 101 - Custom Training Script
============================================

A customizable training script that you can adapt for your own models and datasets.
"""

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import argparse
import os


def train_loop_per_worker(config):
    """
    Customize this function for your specific training needs.
    
    Args:
        config: Dictionary containing hyperparameters and configuration
    """
    from ray.train import get_dataset_shard, report
    from ray.train.torch import prepare_model, prepare_data_loader
    
    # Extract configuration
    lr = config.get("lr", 0.001)
    batch_size = config.get("batch_size", 64)
    num_epochs = config.get("num_epochs", 10)
    
    # Get dataset shards
    train_shard = get_dataset_shard("train")
    val_shard = get_dataset_shard("val")
    
    # Convert to PyTorch DataLoaders
    train_loader = train_shard.to_torch(batch_size=batch_size)
    val_loader = val_shard.to_torch(batch_size=batch_size)
    
    # Prepare for distributed training
    train_loader = prepare_data_loader(train_loader)
    val_loader = prepare_data_loader(val_loader)
    
    # TODO: Define your model here
    # model = YourModel(...)
    # model = prepare_model(model)
    
    # TODO: Define your loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        # TODO: Implement your training logic
        # model.train()
        # for batch in train_loader:
        #     # Training step
        #     pass
        
        # TODO: Implement validation
        # model.eval()
        # with torch.no_grad():
        #     for batch in val_loader:
        #         # Validation step
        #         pass
        
        # Report metrics
        metrics = {
            "epoch": epoch + 1,
            # Add your metrics here
        }
        report(metrics)


def main():
    parser = argparse.ArgumentParser(description="Ray ML Pipeline 101 - Custom Training")
    parser.add_argument("--address", type=str, default=None,
                        help="Ray cluster address (e.g., ray://head-node:10001)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of training workers")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Initialize Ray
    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(ignore_reinit_error=True)
    
    print(f"Ray initialized: {ray.is_initialized()}")
    print(f"Available resources: {ray.available_resources()}")
    
    # Configuration
    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
    }
    
    # Scaling configuration
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
    )
    
    # Run configuration
    run_config = RunConfig(
        storage_path="./ray_results",
        name="custom_training",
    )
    
    # TODO: Load your datasets
    # train_dataset = load_your_train_dataset()
    # val_dataset = load_your_val_dataset()
    
    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        # datasets={"train": train_dataset, "val": val_dataset},
    )
    
    # Run training
    print("Starting training...")
    result = trainer.fit()
    
    print("\nTraining Complete!")
    print(f"Results: {result.metrics}")
    print(f"Checkpoint: {result.checkpoint}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()

