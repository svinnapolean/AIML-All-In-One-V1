"""
FastAPI Backend for Numerics Processor

This module provides REST API endpoints for:
- Model inference and predictions
- AI agent interaction
- Data analysis and visualization
- Model management
"""

from .main import app
from .routers import models, agent, data, health
from .middleware import auth, logging

__all__ = [
    "app",
    "models",
    "agent", 
    "data",
    "health",
    "auth",
    "logging"
]