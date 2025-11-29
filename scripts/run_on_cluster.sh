#!/bin/bash
# Run a Python script on the Ray cluster
# Usage: ./run_on_cluster.sh <script.py> [args...]

set -e

SCRIPT=${1:-}
if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script not found: $SCRIPT"
    exit 1
fi

# Get the head pod name
HEAD_POD=$(kubectl get pods -n default -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$HEAD_POD" ]; then
    echo "Error: Ray head pod not found"
    exit 1
fi

echo "🚀 Running $SCRIPT on Ray cluster..."
echo "   Head pod: $HEAD_POD"

# Copy script to cluster
echo "📋 Copying script to cluster..."
kubectl cp "$SCRIPT" default/$HEAD_POD:/tmp/$(basename "$SCRIPT") -c ray-head

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    kubectl exec -n default $HEAD_POD -c ray-head -- \
        pip install -q -r /tmp/requirements.txt 2>&1 | grep -v "already satisfied" || true
fi

# Run script with remaining arguments
echo "▶️  Executing script..."
shift
kubectl exec -n default $HEAD_POD -c ray-head -- \
    python /tmp/$(basename "$SCRIPT") "$@"

echo "✅ Done!"

