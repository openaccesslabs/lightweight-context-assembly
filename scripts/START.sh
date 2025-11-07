#!/bin/bash
# Quick start script for memory proxy

cd "$(dirname "$0")/.." || exit 1

echo "=================================================="
echo "Memory Proxy Quick Start"
echo "=================================================="
echo ""

# Check if llama-server is running
if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "✓ llama-server is already running on port 8080"
else
    echo "✗ llama-server is NOT running on port 8080"
    echo ""
    echo "Please start llama-server in another terminal:"
    echo ""
    echo "  llama-server -hf ggml-org/gemma-3-1b-it-GGUF --embedding --pooling mean --port 8080 --ctx-size 8192"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check embeddings endpoint
echo "Testing embeddings endpoint..."
if curl -s -X POST http://localhost:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"input":"test"}' | grep -q "embedding"; then
    echo "✓ Embeddings endpoint working"
else
    echo "✗ Embeddings endpoint not working"
    echo "Make sure llama-server was started with --embedding and --pooling mean flags"
    exit 1
fi

echo ""
echo "✓ All checks passed!"
echo ""
echo "Starting memory proxy on port 8081..."
echo "Open your browser to: http://localhost:8081/"
echo ""

python3 run.py
