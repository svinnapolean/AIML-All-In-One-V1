#!/bin/bash
# API Startup Script

set -e

echo "🚀 Starting Loan Default Prediction API..."

# Validate environment
echo "📋 Environment Configuration:"
echo "  - Python Path: $PYTHONPATH"
echo "  - Model Path: $MODEL_PATH"
echo "  - API Host: $API_HOST"
echo "  - API Port: $API_PORT"
echo "  - Log Level: $LOG_LEVEL"

# Check if models directory exists and has models
if [ -d "/app/src/models/saved_models" ]; then
    model_count=$(find /app/src/models/saved_models -name "*.pkl" -o -name "*.h5" 2>/dev/null | wc -l)
    echo "  - Available Models: $model_count"
    
    if [ $model_count -eq 0 ]; then
        echo "⚠️  Warning: No trained models found. API will start but predictions may fail."
        echo "   To fix this, mount your trained models to /app/src/models/saved_models"
    fi
else
    echo "⚠️  Warning: Models directory not found. Creating..."
    mkdir -p /app/src/models/saved_models
fi

# Check if loan data exists
if [ ! -d "/app/loan_data" ]; then
    echo "⚠️  Warning: Loan data directory not found. Some features may not work."
    mkdir -p /app/loan_data
fi

# Start the API
echo "🎯 Starting FastAPI server..."
cd /app

exec python -c "
import sys
sys.path.append('.')
sys.path.append('./src')

# Direct import from model_api to avoid dependency issues
from src.api.model_api import app
import uvicorn

uvicorn.run(
    app, 
    host='${API_HOST}', 
    port=${API_PORT}, 
    log_level='${LOG_LEVEL,,}', 
    access_log=True
)"