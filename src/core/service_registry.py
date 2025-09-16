# Service registry for inter-service communication
import httpx
from .dependency_manager import dependency_manager
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ServiceRegistry:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    def _base_url(self, service_name: str) -> str:
        svc = dependency_manager.services.get(service_name)
        if not svc:
            raise ValueError(f"Unknown service: {service_name}")
        return svc.url.rstrip("/")
    
    async def call_service(self, service_name: str, endpoint: str, 
                          method: str = "GET", **kwargs):
        try:
            url = f"{self._base_url(service_name)}{endpoint}"
            resp = await self._client.request(method.upper(), url, **kwargs)
            resp.raise_for_status()
            return resp.json()
                
        except Exception as e:
            logger.error(f"Error calling {service_name}{endpoint}: {e}")
            return None
    
    async def health_check_all(self) -> Dict[str, str]:
        """Check health of all services"""
        results = {}
        for service_name in self.services:
            try:
                health = await self.call_service(service_name, "/health")
                results[service_name] = health.get("status", "unknown") if health else "unhealthy"
            except Exception as e:
                results[service_name] = "error"
        return results

# Global service registry
service_registry = ServiceRegistry()
