#!/bin/bash

# Startup script for the Tokyo Rent Predictor API (Updated for new structure)

echo "🚀 Starting Tokyo Rent Predictor API..."
echo "==============================================="

# Check if we're in the API directory
if [ ! -f "model_api.py" ]; then
    echo "❌ Error: model_api.py not found. Please run this script from the api directory."
    exit 1
fi

# Check if models exist in the new location
if [ ! -f "models/passed-model.joblib" ]; then
    echo "⚠️  Warning: models/passed-model.joblib not found"
else
    echo "✅ Found passed model"
fi

if [ ! -f "models/challenger-model.joblib" ]; then
    echo "⚠️  Warning: models/challenger-model.joblib not found"
else
    echo "✅ Found challenger model"
fi

# Check if required directories exist
for dir in "models" "services" "utils" "tests"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️  Warning: $dir directory not found"
    else
        echo "✅ Found $dir directory"
    fi
done

# Set environment variables if not set
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

echo ""
echo "🌐 Starting API server..."
echo "   Host: ${HOST}"
echo "   Port: ${PORT}"
echo "   Logs: Check console output"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Start the API using uv
echo "🔄 Using uv to run the application..."
uv run model_api.py