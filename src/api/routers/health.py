"""
Health Check Router

Provides health check and system status endpoints
"""

import os
import psutil
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    uptime: str
    system_info: Dict[str, Any]


class SystemStatus(BaseModel):
    """System status response model"""
    status: str
    services: Dict[str, str]
    resources: Dict[str, Any]
    disk_usage: Dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime=str(datetime.now() - datetime.fromtimestamp(psutil.boot_time())),
            system_info=system_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/status", response_model=SystemStatus)
async def system_status():
    """Detailed system status"""
    try:
        # Check services
        services = {
            "api": "running",
            "models_directory": "available" if os.path.exists("models") else "missing",
            "data_directory": "available" if os.path.exists("data") else "missing",
            "results_directory": "available" if os.path.exists("results") else "missing"
        }
        
        # Resource usage
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        resources = {
            "cpu_cores": cpu_count,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_percent": memory.percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else "N/A"
        }
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent
        }
        
        # Determine overall status
        status = "healthy"
        if memory.percent > 90 or disk.percent > 90:
            status = "warning"
        if memory.percent > 95 or disk.percent > 95:
            status = "critical"
        
        return SystemStatus(
            status=status,
            services=services,
            resources=resources,
            disk_usage=disk_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check"""
    try:
        # Check if essential directories exist
        required_dirs = ["models", "data", "results"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Check system resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        if memory.percent > 95:
            raise HTTPException(status_code=503, detail="Memory usage too high")
        
        if disk.percent > 95:
            raise HTTPException(status_code=503, detail="Disk usage too high")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")


@router.get("/live")
async def liveness_check():
    """Kubernetes-style liveness check"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}