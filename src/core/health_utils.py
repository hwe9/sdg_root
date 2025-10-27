# src/core/health_utils.py
import logging
from datetime import datetime
from typing import Dict
from typing import Any
from typing import Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class HealthCheckResponse:
    """Standardized health check response format"""
    
    @staticmethod
    def healthy_response(
        service_name: str,
        version: str,
        components: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate standardized healthy response"""
        return {
            "status": "healthy",
            "service": service_name,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components or {},
            "dependencies": dependencies or {}
        }
    
    @staticmethod
    def unhealthy_response(
        service_name: str,
        version: str,
        error: Optional[str] = None,
        components: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate standardized unhealthy response"""
        response = {
            "status": "unhealthy",
            "service": service_name,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components or {},
            "dependencies": dependencies or {}
        }
        if error:
            response["error"] = error
        return response
    
    @staticmethod
    def error_response(
        service_name: str,
        version: str,
        error: str
    ) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            "status": "error",
            "service": service_name,
            "version": version,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }

def create_health_endpoint(service_name: str, version: str):
    """Decorator to create standardized health check endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                if result.get("status") == "healthy":
                    return JSONResponse(
                        status_code=200,
                        content=HealthCheckResponse.healthy_response(
                            service_name, version, 
                            result.get("components"),
                            result.get("dependencies")
                        )
                    )
                else:
                    return JSONResponse(
                        status_code=503,
                        content=HealthCheckResponse.unhealthy_response(
                            service_name, version,
                            result.get("error"),
                            result.get("components"),
                            result.get("dependencies")
                        )
                    )
            except Exception as e:
                logger.error(f"Health check error in {service_name}: {e}")
                return JSONResponse(
                    status_code=503,
                    content=HealthCheckResponse.error_response(service_name, version, str(e))
                )
        return wrapper
    return decorator

