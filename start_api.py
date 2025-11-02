#!/usr/bin/env python3
"""
Start the Model API Server
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.api.model_api import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Model API Server...")
    print("📁 Project root:", project_root)
    print("🌐 Server will be available at: http://localhost:8003")
    print("📖 API documentation at: http://localhost:8003/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8003, 
        log_level="info",
        reload=False  # Disable reload to prevent issues
    )