"""
Ray ML Pipeline 101 - Main Pipeline
====================================

This script demonstrates a complete ML pipeline using Ray for distributed training.
It includes data loading, preprocessing, distributed training, and evaluation.
"""

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for classification."""
    
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(n_samples=10000, n_features=20, random_state=42):
    """Generate synthetic classification dataset."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # Create a simple classification target
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
    return X, y


def train_loop_per_worker(config):
    """
    Training function executed on each worker.
    This function is called by Ray Train for distributed training.
    """
    from ray.train import get_dataset_shard, report
    from ray.train.torch import prepare_model, prepare_data_loader
    
    # Get hyperparameters from config
    lr = config.get("lr", 0.001)
    batch_size = config.get("batch_size", 64)
    num_epochs = config.get("num_epochs", 10)
    hidden_dim = config.get("hidden_dim", 64)
    input_dim = config.get("input_dim", 20)
    num_classes = config.get("num_classes", 2)
    
    # Get the dataset shard for this worker
    train_shard = get_dataset_shard("train")
    val_shard = get_dataset_shard("val")
    
    # Convert Ray Dataset to PyTorch DataLoader
    # When label_column is specified, to_torch returns (features, labels) tuples
    train_loader = train_shard.to_torch(
        batch_size=batch_size,
        label_column="labels",
        feature_columns="features"
    )
    val_loader = val_shard.to_torch(
        batch_size=batch_size,
        label_column="labels",
        feature_columns="features"
    )
    
    # Prepare data loaders for distributed training
    train_loader = prepare_data_loader(train_loader)
    val_loader = prepare_data_loader(val_loader)
    
    # Create model
    model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model = prepare_model(model)  # Prepare model for distributed training
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Ray to_torch with label_column returns (features, labels) tuple
            inputs, labels = batch
            # Ensure inputs are float32 and labels are long
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Ray to_torch with label_column returns (features, labels) tuple
                inputs, labels = batch
                # Ensure inputs are float32 and labels are long
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Report metrics to Ray Train
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
        }
        report(metrics)
        
        # Print progress (only on rank 0)
        if ray.train.get_context().get_world_rank() == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")


def create_ray_datasets(X_train, y_train, X_val, y_val):
    """Convert numpy arrays to Ray Datasets."""
    import ray.data
    
    # Create Ray Datasets from numpy arrays
    # Ray Data expects dictionaries with feature and label keys
    train_items = [
        {"features": feat.tolist() if isinstance(feat, np.ndarray) else list(feat), 
         "labels": int(label)} 
        for feat, label in zip(X_train, y_train)
    ]
    
    val_items = [
        {"features": feat.tolist() if isinstance(feat, np.ndarray) else list(feat), 
         "labels": int(label)} 
        for feat, label in zip(X_val, y_val)
    ]
    
    # Create Ray Datasets
    train_dataset = ray.data.from_items(train_items)
    val_dataset = ray.data.from_items(val_items)
    
    return train_dataset, val_dataset


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("Ray ML Pipeline 101")
    print("=" * 60)
    
    # Initialize Ray
    # For local testing: ray.init()
    # For cluster: ray.init(address="ray://<head-node>:10001")
    print("\n1. Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    print(f"   Ray initialized: {ray.is_initialized()}")
    print(f"   Available resources: {ray.available_resources()}")
    
    # Generate synthetic data
    print("\n2. Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=10000, n_features=20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Create Ray Datasets
    print("\n3. Creating Ray Datasets...")
    train_dataset, val_dataset = create_ray_datasets(
        X_train, y_train, X_val, y_val
    )
    
    # Training configuration
    config = {
        "lr": 0.001,
        "batch_size": 64,
        "num_epochs": 10,
        "hidden_dim": 64,
        "input_dim": 20,
        "num_classes": 2,
    }
    
    # Scaling configuration
    # Adjust num_workers based on your cluster
    scaling_config = ScalingConfig(
        num_workers=2,  # Use 2 workers for distributed training
        use_gpu=False,  # Set to True if GPUs are available
    )
    
    # Run configuration
    run_config = RunConfig(
        storage_path="./ray_results",
        name="ml_pipeline_101",
    )
    
    # Create trainer
    print("\n4. Setting up Ray Trainer...")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset, "val": val_dataset},
    )
    
    # Run training
    print("\n5. Starting distributed training...")
    print("   Monitor progress in Ray Dashboard: http://localhost:8265")
    print("-" * 60)
    
    result = trainer.fit()
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {result.path}")
    print(f"Checkpoint available at: {result.checkpoint}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\nRay shutdown complete.")


if __name__ == "__main__":
    main()

