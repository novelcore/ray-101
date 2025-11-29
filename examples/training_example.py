"""
Example: Distributed Training with Ray Train

This example shows how to train a model across multiple GPUs/nodes
using Ray Train on the cluster.
"""

import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import ray.data


def create_model():
    """Create a simple model"""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def train_loop_per_worker(config):
    """Training function executed on each worker"""
    # Get model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    criterion = nn.MSELoss()
    
    # Get dataset shard for this worker
    train_ds = train.get_dataset_shard("train")
    
    num_epochs = config.get("num_epochs", 10)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Iterate over batches
        for batch in train_ds.iter_torch_batches(batch_size=32, dtypes=torch.float32):
            # Get inputs and targets
            inputs = batch["features"]
            targets = batch["target"]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        # Report metrics
        train.report({
            "loss": avg_loss,
            "epoch": epoch
        })
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = Checkpoint.from_dict({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss
            })
            train.report({"loss": avg_loss}, checkpoint=checkpoint)


def main():
    # Connect to Ray cluster
    # Option 1: Via port-forward (local development)
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True)
    
    # Option 2: Direct connection (if you have cluster access)
    # ray.init(address="ray://<head-node-ip>:10001")
    
    print("✅ Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")
    
    # Create sample dataset
    # In practice, load your actual dataset
    data = [
        {
            "features": torch.randn(10).tolist(),
            "target": torch.randn(1).item()
        }
        for _ in range(1000)
    ]
    
    train_dataset = ray.data.from_items(data)
    
    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "num_epochs": 10,
            "lr": 0.001
        },
        scaling_config=ScalingConfig(
            num_workers=2,  # Use 2 workers
            use_gpu=True     # Use GPUs if available
        ),
        datasets={"train": train_dataset}
    )
    
    # Train
    print("🚀 Starting training...")
    result = trainer.fit()
    
    print(f"\n✅ Training complete!")
    print(f"Best metrics: {result.metrics}")
    print(f"Checkpoints: {len(result.best_checkpoints)}")
    
    # Access best checkpoint
    if result.best_checkpoints:
        best_checkpoint = result.best_checkpoints[0][1]
        print(f"Best checkpoint path: {best_checkpoint.path}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()

