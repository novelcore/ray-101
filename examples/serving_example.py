"""
Example: Model Serving with Ray Serve

This example shows how to deploy a model as an HTTP service
using Ray Serve on the cluster.
"""

import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import torch
import torch.nn as nn


# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


app = FastAPI()


@serve.deployment(
    num_replicas=2,
    ray_actor_options={
        "num_gpus": 1,  # Use GPU
        "num_cpus": 2
    }
)
@serve.ingress(app)
class ModelServer:
    """Model serving deployment"""
    
    def __init__(self):
        print("🔄 Loading model...")
        self.model = SimpleModel()
        self.model.eval()
        print("✅ Model loaded")
    
    @app.get("/health")
    async def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "model-server"
        }
    
    @app.post("/predict")
    async def predict(self, request: Dict[str, Any]):
        """
        Prediction endpoint
        
        Request body:
        {
            "features": [0.1, 0.2, ...]  # 10 features
        }
        """
        try:
            features = request.get("features")
            if not features or len(features) != 10:
                raise HTTPException(
                    status_code=400,
                    detail="Expected 10 features"
                )
            
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(input_tensor)
            
            return {
                "prediction": prediction.item(),
                "features": features
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def main():
    # Connect to Ray cluster
    # Option 1: Via port-forward (local development)
    ray.init(address="ray://localhost:10001", ignore_reinit_error=True)
    
    # Option 2: Direct connection
    # ray.init(address="ray://<head-node-ip>:10001")
    
    # Option 3: Auto-connect when running inside cluster
    # ray.init(ignore_reinit_error=True)
    
    print("✅ Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")
    
    # Deploy service
    print("🚀 Deploying model service...")
    serve.run(ModelServer.bind(), route_prefix="/model")
    
    print("\n✅ Service deployed!")
    print("📊 Endpoints:")
    print("  - Health: http://localhost:8000/model/health")
    print("  - Predict: http://localhost:8000/model/predict")
    print("\n🧪 Test with:")
    print('  curl -X POST "http://localhost:8000/model/predict" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}\'')
    print("\n⏹️  Press Ctrl+C to stop")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()

