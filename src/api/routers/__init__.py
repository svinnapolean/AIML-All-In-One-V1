"""
API Routers Package
"""

from .models import router as models_router
from .agent import router as agent_router
from .data import router as data_router
from .health import router as health_router

__all__ = ["models_router", "agent_router", "data_router", "health_router"]