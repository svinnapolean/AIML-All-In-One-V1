"""
AI/ML Learning Platform - FastAPI Main Application

This module sets up the main FastAPI application for the AI/ML Learning Platform with:
- Educational API endpoints for ML model training and deployment
- AI agent interaction for learning agent development
- Public API endpoints for ML model consumption
- Middleware for authentication, CORS, and logging
- Global error handling and monitoring
- WebSocket support for real-time agent interaction

Educational Focus:
- Demonstrates production-ready ML API development
- Shows best practices for ML model serving
- Integrates Azure ADK evaluation for AI systems
- Provides examples of CI/CD for ML applications
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import routers
from .routers.models import router as models_router
from .routers.agent import router as agent_router
from .routers.data import router as data_router
from .routers.health import router as health_router

# Import middleware
from .middleware.auth import AuthMiddleware
from .middleware.logging import LoggingMiddleware

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Numerics Processor API")
    
    # Create necessary directories
    for directory in ['models', 'data', 'results', 'uploads']:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Numerics Processor API")


# Create FastAPI application
app = FastAPI(
    title="Numerics Processor API",
    description="""
    A comprehensive AI-powered API for numerical data processing and machine learning.
    
    ## Features
    
    * **Model Management**: Train, test, and evaluate machine learning models
    * **AI Agent**: Intelligent assistant for data analysis and insights
    * **Data Processing**: Statistical analysis and visualization tools
    * **Real-time Interaction**: WebSocket support for streaming responses
    
    ## Getting Started
    
    1. Upload your data using the `/data/upload` endpoint
    2. Train a model using the `/models/train` endpoint
    3. Interact with the AI agent via `/agent/chat` or WebSocket
    4. Get predictions using `/models/predict`
    
    ## Authentication
    
    Some endpoints require API key authentication. Include your API key in the header:
    `Authorization: Bearer your-api-key`
    """,
    version="1.0.0",
    contact={
        "name": "AI SimpleLearn",
        "email": "admin@simplelearn.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add CORS middleware
allowed_origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "http://localhost:3001",  # Docker frontend port
    "http://127.0.0.1:3001"
]

# In production, allow all origins if behind nginx proxy
if os.getenv("ENVIRONMENT") == "production":
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(models_router, prefix="/models", tags=["Models"])
app.include_router(agent_router, prefix="/agent", tags=["AI Agent"])
app.include_router(data_router, prefix="/data", tags=["Data"])

# Serve static files (for frontend if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Numerics Processor API",
        "version": "1.0.0",
        "description": "AI-powered numerical data processing and machine learning API",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "agent": "/agent", 
            "data": "/data",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "Machine Learning Model Training & Evaluation",
            "AI Agent for Data Analysis",
            "Statistical Data Processing",
            "Real-time WebSocket Communication",
            "Comprehensive API Documentation"
        ]
    }


# API Info endpoint
@app.get("/info", tags=["Root"])
async def api_info():
    """Get detailed API information"""
    return {
        "api_name": "Numerics Processor",
        "version": "1.0.0",
        "framework": "FastAPI",
        "python_version": "3.9+",
        "dependencies": {
            "fastapi": "Latest",
            "uvicorn": "Latest", 
            "pandas": "2.1.0+",
            "scikit-learn": "1.3.0+",
            "microsoft-agent-framework": "Preview"
        },
        "supported_models": [
            "Linear Regression",
            "Random Forest",
            "Neural Networks",
            "Custom Models"
        ],
        "ai_models": [
            "GitHub Models (GPT-4.1, GPT-4o, etc.)",
            "Azure AI Foundry Models"
        ]
    }


@app.get("/models", tags=["Root"])
async def get_available_models():
    """Get available trained models for quick access"""
    try:
        import os
        import joblib
        from datetime import datetime
        
        models_path = "models/trained_models"
        models = {}
        
        # Check if models directory exists
        if os.path.exists(models_path):
            for file in os.listdir(models_path):
                if file.endswith('.joblib'):
                    model_name = file.replace('.joblib', '')
                    model_path = os.path.join(models_path, file)
                    
                    # Get file stats
                    stat = os.stat(model_path)
                    created_date = datetime.fromtimestamp(stat.st_ctime)
                    
                    # Try to load model to get type
                    try:
                        model = joblib.load(model_path)
                        model_type = type(model).__name__
                    except:
                        model_type = "Unknown"
                    
                    models[model_name] = {
                        "type": model_type,
                        "created": created_date.isoformat(),
                        "size_kb": round(stat.st_size / 1024, 2),
                        "performance": {
                            "roc_auc": 0.85,  # Mock data
                            "accuracy": 0.82
                        }
                    }
        
        # Add default sample models if none exist
        if not models:
            from datetime import datetime as dt
            models = {
                "sample-classifier": {
                    "type": "RandomForestClassifier",
                    "created": dt.now().isoformat(),
                    "size_kb": 156.7,
                    "performance": {"roc_auc": 0.85, "accuracy": 0.82}
                },
                "sample-regressor": {
                    "type": "RandomForestRegressor", 
                    "created": dt.now().isoformat(),
                    "size_kb": 203.4,
                    "performance": {"r2_score": 0.78, "mse": 0.23}
                }
            }
        
        return {
            "available_models": models,
            "total_models": len(models),
            "models_directory": models_path
        }
        
    except Exception as e:
        from datetime import datetime as dt
        return {
            "available_models": {
                "sample-classifier": {
                    "type": "RandomForestClassifier",
                    "created": dt.now().isoformat(),
                    "size_kb": 156.7,
                    "performance": {"roc_auc": 0.85, "accuracy": 0.82}
                },
                "sample-regressor": {
                    "type": "RandomForestRegressor", 
                    "created": dt.now().isoformat(),
                    "size_kb": 203.4,
                    "performance": {"r2_score": 0.78, "mse": 0.23}
                }
            },
            "total_models": 2,
            "error": f"Unable to load models: {str(e)}"
        }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )