"""
Example: Fine-tuning with Ray Train

This example shows how to fine-tuning a model using Ray Train
on the cluster with distributed training.
"""

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import ray.data
import os


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


@ray.remote
def create_dataset():
    """Create dataset on the cluster side to avoid Ray Client serialization issues"""
    import ray.data
    import torch
    
    fine_tune_data = [
        {
            "features": torch.randn(10).tolist(),
            "target": torch.randn(1).item()
        }
        for _ in range(500)  # Smaller dataset for fine-tuning
    ]
    
    return ray.data.from_items(fine_tune_data)


def fine_tune_loop(config):
    """Fine-tuning function executed on each worker"""
    # Get the device from Ray Train
    device = train.torch.get_device()
    print(f"Training on device: {device}")
    
    # Load base model
    model = load_pretrained_model(config)
    
    # Freeze some layers (optional)
    if config.get("freeze_base", False):
        for param in list(model.parameters())[:-2]:  # Freeze all but last 2 layers
            param.requires_grad = False
    
    # Prepare model (handles device placement + DDP wrapping)
    model = train.torch.prepare_model(model)
    
    # Setup optimizer AFTER preparing model (only train unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.get("lr", 1e-4))
    criterion = nn.MSELoss()
    
    # Get fine-tuning dataset
    train_ds = train.get_dataset_shard("train")
    
    num_epochs = config.get("num_epochs", 5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Use device parameter to move tensors to correct device
        for batch in train_ds.iter_torch_batches(
            batch_size=32, 
            dtypes=torch.float32,
            device=device  # This automatically moves tensors to the correct device
        ):
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
        metrics = {
            "loss": avg_loss,
            "epoch": epoch
        }
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            # Get underlying model state (unwrap from DDP if needed)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Create a persistent checkpoint directory
            checkpoint_dir = f"/tmp/checkpoint_epoch_{epoch}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            try:
                # Save model state
                torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))
                
                # Save optimizer state
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                
                # Save metadata
                import json
                metadata = {
                    "epoch": epoch,
                    "loss": avg_loss
                }
                with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f)
                
                # Create checkpoint from directory - Ray Train will copy this
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(metrics, checkpoint=checkpoint)
            finally:
                # Clean up after Ray Train has copied the checkpoint
                # (Ray Train copies the directory contents, so we can clean up)
                import shutil
                import time
                time.sleep(1)  # Give Ray Train time to copy
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            train.report(metrics)


def main():
    # Connect to Ray cluster
    runtime_env = {
        "pip": ["torch>=2.0.0", "numpy>=1.24.0"]
    }
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True, runtime_env=runtime_env)
    
    print("✅ Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")
    
    # Create fine-tuning dataset on the cluster (not locally)
    print("📊 Creating dataset on cluster...")
    fine_tune_dataset = ray.get(create_dataset.remote())
    print("✅ Dataset created")
    
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
            num_workers=1,  # Use 1 worker (cluster has 1 GPU)
            use_gpu=True    # Use GPU
        ),
        datasets={"train": fine_tune_dataset},
        run_config=RunConfig(
            name="finetuning_example",
            storage_path="/tmp/ray_results",
        ),
    )
    
    # Fine-tune
    print("🚀 Starting fine-tuning...")
    result = trainer.fit()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Final metrics: {result.metrics}")
    
    # Access checkpoint
    if result.checkpoint:
        best_checkpoint = result.checkpoint
        print(f"Best checkpoint path: {best_checkpoint.path}")
        
        try:
            # Load fine-tuned model from checkpoint
            with best_checkpoint.as_directory() as checkpoint_dir:
                print(f"Loading from checkpoint directory: {checkpoint_dir}")
                
                # List files in checkpoint directory for debugging
                files = os.listdir(checkpoint_dir)
                print(f"Files in checkpoint: {files}")
                
                model = load_pretrained_model({"pretrained_path": None})
                model_path = os.path.join(checkpoint_dir, "model.pt")
                
                if os.path.exists(model_path):
                    model_state = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(model_state)
                    
                    # Load metadata if it exists
                    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        print(f"✅ Fine-tuned model loaded from epoch {metadata['epoch']} with loss {metadata['loss']:.4f}")
                    else:
                        print("✅ Fine-tuned model loaded (no metadata found)")
                else:
                    print(f"⚠️ Model file not found at {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ No checkpoint available")
    
    ray.shutdown()


if __name__ == "__main__":
    main()