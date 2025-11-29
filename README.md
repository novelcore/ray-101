# Ray ML Pipeline 101

A beginner-friendly guide to building distributed ML pipelines with Ray.

## Overview

This repository demonstrates how to build and run distributed machine learning pipelines using Ray. It includes examples for:

- Distributed PyTorch training
- Data preprocessing with Ray Data
- Model evaluation and checkpointing
- Connecting to a Ray cluster

## Prerequisites

- Python 3.8+
- Access to a Ray cluster (or run locally with `ray start --head`)
- Required Python packages (see `requirements.txt`)

## Ray Cluster Connection

Your Ray cluster is configured with:
- **Ray Version**: 2.10.0
- **Dashboard**: Available at port 8265
- **GPU Support**: Enabled (NVIDIA A6000)
- **Metrics**: Exposed at port 8080

### Connecting to the Cluster

```python
import ray

# Connect to your Ray cluster
ray.init(address="ray://<head-node-address>:10001")
```

Or if running locally:
```python
ray.init()
```

## Project Structure

```
ray-101/
├── README.md
├── requirements.txt
├── pipeline.py          # Main ML pipeline
├── train.py             # Training script
└── utils/
    └── data_loader.py   # Data loading utilities
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python pipeline.py
```

3. Monitor training:
   - Ray Dashboard: `http://<ray-head>:8265`
   - Check logs for training progress

## Examples

### Basic Training Pipeline

See `pipeline.py` for a complete example of:
- Loading and preprocessing data
- Distributed model training
- Model evaluation
- Saving checkpoints

### Custom Training

See `train.py` for a customizable training script that you can adapt for your own models.

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Train Guide](https://docs.ray.io/en/master/train/train.html)
- [Ray Data Guide](https://docs.ray.io/en/master/data/data.html)

## License

MIT

