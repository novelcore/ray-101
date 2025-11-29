# Connecting to Your Ray Cluster

This guide explains how to connect to your company's Ray cluster for distributed ML training.

## Cluster Information

Based on your infrastructure configuration:

- **Ray Version**: 2.10.0
- **Cluster Name**: `novelcore-private-ray-cluster`
- **Namespace**: `default`
- **Dashboard Port**: 8265
- **Client Port**: 10001
- **GPU Support**: Enabled (NVIDIA A6000)

## Finding Your Ray Cluster Address

### Option 1: Using Kubernetes Service

The Ray head node is accessible via Kubernetes service. To find the service address:

```bash
# Get the Ray head service
kubectl get svc -n default | grep ray

# The service name will be something like:
# novelcore-private-ray-cluster-head-svc
```

### Option 2: Port Forwarding (For Local Development)

If you're working from your local machine, you can use port forwarding:

```bash
# Forward the Ray client port
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc 10001:10001

# Forward the Ray dashboard
kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc 8265:8265
```

Then connect using:
```python
ray.init(address="ray://localhost:10001")
```

### Option 3: Direct Cluster Access

If you have direct network access to the cluster:

```python
# Replace <head-node-ip> with your actual head node IP
ray.init(address="ray://<head-node-ip>:10001")
```

## Connection Examples

### Local Development (No Cluster)

For testing without connecting to the cluster:

```python
import ray

# Start Ray locally
ray.init()
```

### Connect to Remote Cluster

```python
import ray

# Connect to your Ray cluster
ray.init(
    address="ray://<head-node-address>:10001",
    runtime_env={
        "pip": ["torch>=2.0.0", "numpy>=1.24.0"]
    }
)
```

### Verify Connection

```python
import ray

ray.init(address="ray://<head-node-address>:10001")

# Check cluster status
print(f"Ray initialized: {ray.is_initialized()}")
print(f"Available resources: {ray.available_resources()}")
print(f"Cluster resources: {ray.cluster_resources()}")

# Check GPU availability
if "GPU" in ray.available_resources():
    print(f"GPUs available: {ray.available_resources()['GPU']}")
```

## Accessing the Ray Dashboard

The Ray Dashboard provides a web UI for monitoring your cluster and jobs:

1. **Via Port Forwarding**:
   ```bash
   kubectl port-forward -n default svc/novelcore-private-ray-cluster-head-svc 8265:8265
   ```
   Then open: `http://localhost:8265`

2. **Direct Access** (if network allows):
   `http://<head-node-ip>:8265`

## Running the Pipeline

Once connected, you can run the ML pipeline:

```bash
# Make sure you're connected to the cluster
python pipeline.py
```

Or modify `pipeline.py` to connect to your cluster:

```python
# In pipeline.py, change:
ray.init(ignore_reinit_error=True)

# To:
ray.init(address="ray://<your-cluster-address>:10001")
```

## Troubleshooting

### Connection Refused

- Verify the Ray cluster is running: `kubectl get pods -n default | grep ray`
- Check service is accessible: `kubectl get svc -n default | grep ray`
- Verify network connectivity to the cluster

### Authentication Issues

- Ensure you have proper Kubernetes access
- Check if your cluster requires authentication tokens

### GPU Not Available

- Verify GPU resources: `ray.available_resources()`
- Check if GPU workers are running: `kubectl get pods -n default | grep ray-worker`
- Ensure your code requests GPU: `use_gpu=True` in `ScalingConfig`

## Next Steps

1. Test the connection with a simple script
2. Run the example pipeline
3. Monitor training in the Ray Dashboard
4. Scale up to use GPU workers for larger models

