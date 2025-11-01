"""
Authentication Middleware

Provides API key-based authentication for protected endpoints
"""

import os
import uuid
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API key validation"""
    
    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key or os.getenv("API_KEY", "dev-api-key-123")
        
        # Endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/health/live", 
            "/health/ready",
            "/health/status",
            "/info",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Add request ID for tracking
        request.state.request_id = str(uuid.uuid4())
        
        # Check if endpoint requires authentication
        path = request.url.path
        
        # Allow public endpoints
        if path in self.public_endpoints or path.startswith("/static"):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        
        # Check for API key in Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing Authorization header",
                    "message": "Include 'Authorization: Bearer your-api-key' in headers",
                    "request_id": request.state.request_id
                }
            )
        
        # Extract API key from Bearer token
        try:
            scheme, credentials = auth_header.split()
            if scheme.lower() != "bearer":
                raise ValueError("Invalid scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid Authorization header format",
                    "message": "Use 'Bearer your-api-key' format",
                    "request_id": request.state.request_id
                }
            )
        
        # Validate API key
        if credentials != self.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid",
                    "request_id": request.state.request_id
                }
            )
        
        # Add user info to request state
        request.state.authenticated = True
        request.state.api_key = credentials
        
        # Process request
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response