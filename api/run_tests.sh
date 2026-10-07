#!/bin/bash

# Test runner script for the Tokyo Rent Predictor API

echo "🧪 Running API Tests with UV..."
echo "================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check if we're in the API directory
if [ ! -f "model_api.py" ]; then
    echo "❌ Error: model_api.py not found. Please run this script from the api directory."
    exit 1
fi

echo "📡 Starting API server in background..."
# Start the API in background
uv run model_api.py &
API_PID=$!

# Give the server time to start
echo "⏳ Waiting for server to start..."
sleep 5

# Run the tests
echo "🚀 Running tests..."
uv run tests/test_api.py

# Clean up - stop the API server
echo "🛑 Stopping API server..."
kill $API_PID 2>/dev/null

echo "✅ Test run complete!"