# Ray Cluster Developer Guide

A practical guide for developers to connect to and use the Ray cluster for distributed ML workloads.

## Quick Start

### 1. Connect to Ray Cluster

```python
import ray

# Connect via port-forward (recommended for local development)
ray.init(address="ray://localhost:10001")

# Or connect directly if you have cluster access
# ray.init(address="ray://<head-node-ip>:10001")
```

**Setup port forwarding:**
```bash
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc 10001:10001 8265:8265
```

### 2. Verify Connection

```python
import ray

ray.init(address="ray://localhost:10001")

print(f"Connected: {ray.is_initialized()}")
print(f"Resources: {ray.cluster_resources()}")
print(f"GPUs: {ray.cluster_resources().get('GPU', 0)}")
```

## Cluster Information

- **Ray Version**: 2.10.0
- **Cluster**: `novelcore-private-ray-cluster` (namespace: `default`)
- **Dashboard**: Port 8265 (forward to `localhost:8265`)
- **Client Port**: 10001 (forward to `localhost:10001`)
- **GPUs**: 2x RTX (48GB VRAM each)

## Use Cases

### 1. Distributed Training (Ray Train)

Train models across multiple GPUs/nodes:

```python
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_loop_per_worker(config):
    # Your training logic here
    model = create_model()
    optimizer = create_optimizer(model)
    
    # Get dataset shard for this worker
    train_ds = train.get_dataset_shard("train")
    
    for epoch in range(config["num_epochs"]):
        for batch in train_ds.iter_torch_batches(batch_size=32):
            # Training step
            loss = train_step(model, optimizer, batch)
            train.report({"loss": loss})

# Create trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"num_epochs": 10},
    scaling_config=ScalingConfig(
        num_workers=2,  # Use 2 workers
        use_gpu=True    # Use GPUs
    ),
    datasets={"train": train_dataset}
)

# Train
result = trainer.fit()
print(f"Best metrics: {result.metrics}")
```

**See**: `examples/training_example.py`

### 2. Model Serving (Ray Serve)

Deploy models as HTTP/gRPC services:

```python
import ray
from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment(num_replicas=2)
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        # Load your model
        self.model = load_model()
    
    @app.post("/predict")
    async def predict(self, request: dict):
        # Run inference
        result = self.model.predict(request["data"])
        return {"prediction": result}

# Deploy
serve.run(ModelServer.bind(), route_prefix="/model")
```

**Access**: `http://localhost:8000/model/predict` (after port-forwarding)

**See**: `examples/serving_example.py`

### 3. Data Preprocessing (Ray Data)

Process large datasets in parallel:

```python
import ray.data

# Load data
dataset = ray.data.read_parquet("s3://bucket/data/*.parquet")
# Or from local files
# dataset = ray.data.read_csv("data/*.csv")

# Preprocess
def preprocess_batch(batch):
    # Your preprocessing logic
    batch["feature"] = normalize(batch["feature"])
    return batch

processed = dataset.map_batches(preprocess_batch, batch_size=1000)

# Save
processed.write_parquet("s3://bucket/processed/")
```

**See**: `examples/preprocessing_example.py`

### 4. Fine-tuning

Fine-tune models with Ray Train:

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def fine_tune_loop(config):
    # Load base model
    model = load_pretrained_model(config["base_model"])
    
    # Fine-tuning logic
    train_ds = train.get_dataset_shard("train")
    
    for batch in train_ds.iter_torch_batches():
        # Fine-tuning step
        loss = fine_tune_step(model, batch)
        train.report({"loss": loss})

trainer = TorchTrainer(
    fine_tune_loop,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    train_loop_config={"base_model": "bert-base-uncased"},
    datasets={"train": fine_tune_dataset}
)

result = trainer.fit()
```

**See**: `examples/finetuning_example.py`

## Running on Cluster

### Option 1: Run Locally (Port Forward)

```bash
# Terminal 1: Port forwarding
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc \
  10001:10001 8265:8265

# Terminal 2: Run your script
python your_script.py
```

### Option 2: Run Inside Cluster

```bash
# Copy script to cluster
kubectl cp your_script.py default/<ray-head-pod>:/tmp/ -c ray-head

# Execute inside cluster
kubectl exec -n default <ray-head-pod> -c ray-head -- \
  python /tmp/your_script.py
```

**Helper script**: `scripts/run_on_cluster.sh`

## Examples

Check the `examples/` directory for complete working examples:

- `training_example.py` - Distributed PyTorch training
- `serving_example.py` - Model serving with FastAPI
- `preprocessing_example.py` - Data preprocessing pipeline
- `finetuning_example.py` - Fine-tuning workflow

## Monitoring

### Ray Dashboard

Access the dashboard:
```bash
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc 8265:8265
```

Then open: `http://localhost:8265`

### Check Resources

```python
import ray

ray.init(address="ray://localhost:10001")

# Available resources
print(ray.available_resources())

# Cluster resources
print(ray.cluster_resources())

# GPU info
if "GPU" in ray.cluster_resources():
    print(f"GPUs: {ray.cluster_resources()['GPU']}")
```

## Common Patterns

### Pattern 1: Training with Checkpoints

```python
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

def train_with_checkpoints(config):
    model = create_model()
    
    for epoch in range(config["num_epochs"]):
        # Training...
        loss = train_epoch(model)
        
        # Save checkpoint
        checkpoint = Checkpoint.from_dict({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": loss
        })
        train.report({"loss": loss}, checkpoint=checkpoint)

trainer = TorchTrainer(
    train_with_checkpoints,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
)
result = trainer.fit()

# Access best checkpoint
best_checkpoint = result.best_checkpoints[0][1]
```

### Pattern 2: Multi-Model Serving

```python
from ray import serve

@serve.deployment
class ModelA:
    def predict(self, data):
        return model_a(data)

@serve.deployment
class ModelB:
    def predict(self, data):
        return model_b(data)

@serve.deployment
class Ensemble:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
    
    async def predict(self, data):
        pred_a = await self.model_a.predict.remote(data)
        pred_b = await self.model_b.predict.remote(data)
        return combine(pred_a, pred_b)

# Deploy ensemble
serve.run(
    Ensemble.bind(ModelA.bind(), ModelB.bind()),
    route_prefix="/ensemble"
)
```

### Pattern 3: Distributed Data Pipeline

```python
import ray.data

# Load and process
pipeline = (
    ray.data.read_parquet("input/*.parquet")
    .map_batches(preprocess, batch_size=1000)
    .filter(filter_valid)
    .map_batches(augment, batch_size=500)
    .write_parquet("output/")
)

# Execute
pipeline.fully_executed()
```

## Troubleshooting

### Connection Issues

```python
# Check if Ray is initialized
if not ray.is_initialized():
    ray.init(address="ray://localhost:10001")

# Verify connection
try:
    resources = ray.cluster_resources()
    print(f"Connected! Resources: {resources}")
except Exception as e:
    print(f"Connection failed: {e}")
```

### GPU Not Available

```python
# Check GPU availability
resources = ray.cluster_resources()
if "GPU" not in resources:
    print("No GPUs available")
    print("Available resources:", resources)
else:
    print(f"GPUs available: {resources['GPU']}")
```

### Version Mismatch

If you get version mismatch errors:
- **Solution**: Run your script inside the cluster pod
- Use `scripts/run_on_cluster.sh` to execute inside cluster
- Cluster runs Ray 2.10.0

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Train Guide](https://docs.ray.io/en/master/train/train.html)
- [Ray Serve Guide](https://docs.ray.io/en/master/serve/index.html)
- [Ray Data Guide](https://docs.ray.io/en/master/data/data.html)

## Getting Help

1. Check Ray Dashboard: `http://localhost:8265`
2. Review logs: `kubectl logs -n default <ray-pod>`
3. Check cluster status: `kubectl get pods -n default | grep ray`
