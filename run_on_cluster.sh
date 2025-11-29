#!/bin/bash
# Submit simple pipeline to Ray cluster via kubectl exec
# This works around version mismatch by running inside the cluster

set -e

PIPELINE=${1:-simple_pipeline.py}
TEXT_PROMPT=${2:-"a red cube"}

echo "🚀 Submitting pipeline to Ray cluster..."
echo "   Pipeline: $PIPELINE"
echo "   Prompt: '$TEXT_PROMPT'"

# Get the head pod name dynamically
HEAD_POD=$(kubectl get pods -n default -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Copy pipeline to cluster
echo "📋 Copying pipeline to cluster..."
kubectl cp $PIPELINE default/$HEAD_POD:/tmp/pipeline.py -c ray-head

# Install minimal dependencies (numpy only for simple pipeline)
echo "📦 Installing dependencies..."
kubectl exec -n default $HEAD_POD -c ray-head -- \
  pip install numpy --quiet 2>&1 | grep -v "already satisfied" || true

# Run the pipeline
echo "▶️  Running pipeline on cluster..."
echo "   Dashboard: http://localhost:8266"
echo "-" * 60

kubectl exec -n default $HEAD_POD -c ray-head -- \
  python /tmp/pipeline.py --text "$TEXT_PROMPT" --num-workers 1

echo ""
echo "✅ Pipeline completed! Check dashboard at http://localhost:8266"

