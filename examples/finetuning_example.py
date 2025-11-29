"""
Example: Fine-tuning with Ray Train

This example shows how to fine-tune a model using Ray Train
on the cluster with distributed training.
"""

import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import ray.data


def load_pretrained_model(config):
    """Load a pretrained model"""
    # In practice, load your actual pretrained model
    # model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    
    # For demo, create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Load pretrained weights if available
    if config.get("pretrained_path"):
        model.load_state_dict(torch.load(config["pretrained_path"]))
    
    return model


def fine_tune_loop(config):
    """Fine-tuning function executed on each worker"""
    # Load base model
    model = load_pretrained_model(config)
    
    # Freeze some layers (optional)
    if config.get("freeze_base", False):
        for param in list(model.parameters())[:-2]:  # Freeze all but last 2 layers
            param.requires_grad = False
    
    # Setup optimizer (only train unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.get("lr", 1e-4))
    criterion = nn.MSELoss()
    
    # Get fine-tuning dataset
    train_ds = train.get_dataset_shard("train")
    
    num_epochs = config.get("num_epochs", 5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_ds.iter_torch_batches(batch_size=32, dtypes=torch.float32):
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
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint = Checkpoint.from_dict({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss
            })
            train.report({"loss": avg_loss}, checkpoint=checkpoint)


def main():
    # Connect to Ray cluster
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True)
    
    print("✅ Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")
    
    # Create fine-tuning dataset
    # In practice, load your fine-tuning dataset
    fine_tune_data = [
        {
            "features": torch.randn(10).tolist(),
            "target": torch.randn(1).item()
        }
        for _ in range(500)  # Smaller dataset for fine-tuning
    ]
    
    fine_tune_dataset = ray.data.from_items(fine_tune_data)
    
    # Create trainer for fine-tuning
    trainer = TorchTrainer(
        fine_tune_loop,
        train_loop_config={
            "num_epochs": 5,
            "lr": 1e-4,  # Lower learning rate for fine-tuning
            "freeze_base": False,  # Set to True to freeze base layers
            "pretrained_path": None  # Path to pretrained weights
        },
        scaling_config=ScalingConfig(
            num_workers=2,  # Use 2 workers
            use_gpu=True    # Use GPUs
        ),
        datasets={"train": fine_tune_dataset}
    )
    
    # Fine-tune
    print("🚀 Starting fine-tuning...")
    result = trainer.fit()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Final metrics: {result.metrics}")
    
    # Access best checkpoint
    if result.best_checkpoints:
        best_checkpoint = result.best_checkpoints[0][1]
        print(f"Best checkpoint: {best_checkpoint.path}")
        
        # Load fine-tuned model
        checkpoint_dict = best_checkpoint.to_dict()
        model = load_pretrained_model({"pretrained_path": None})
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        print("✅ Fine-tuned model loaded")
    
    ray.shutdown()


if __name__ == "__main__":
    main()

