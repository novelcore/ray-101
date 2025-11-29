# Quick Start Guide

## 1. Connect to Cluster

```bash
# Terminal 1: Port forward
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc \
  10001:10001 8265:8265
```

```python
# Terminal 2: Your script
import ray
ray.init(address="ray://localhost:10001")
```

## 2. Run Examples

### Training
```bash
python examples/training_example.py
```

### Serving
```bash
python examples/serving_example.py
# Then test: curl http://localhost:8000/model/health
```

### Preprocessing
```bash
python examples/preprocessing_example.py
```

### Fine-tuning
```bash
python examples/finetuning_example.py
```

## 3. Run on Cluster (Alternative)

```bash
./scripts/run_on_cluster.sh examples/training_example.py
```

## 4. Monitor

Open Ray Dashboard: `http://localhost:8265`

## Common Commands

```python
# Check connection
ray.is_initialized()

# Check resources
ray.cluster_resources()
ray.available_resources()

# Check GPUs
ray.cluster_resources().get('GPU', 0)
```

