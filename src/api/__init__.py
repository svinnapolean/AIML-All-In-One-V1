"""
AI/ML Learning Platform - FastAPI Backend

This module provides educational REST API endpoints for learning:
- ML model training, testing, and inference
- AI agent interaction and development
- Data analysis and visualization
- Azure ADK evaluation integration
- Production deployment practices
- CI/CD for machine learning

Educational Focus:
- Demonstrates modern API development with FastAPI
- Shows ML model serving best practices
- Integrates AI agent development patterns
- Provides examples of ML system evaluation
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