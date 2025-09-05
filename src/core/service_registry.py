# Service registry for inter-service communication
import httpx
import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ServiceRegistry:
    def __init__(self):
        self.services = {
            "api": "http://api_service:8000",
            "data_retrieval": "http://data_retrieval_service:8002", 
            "data_processing": "http://data_processing_service:8001",
            "vectorization": "http://vectorization_service:8003",
            "content_extraction": "http://content_extraction_service:8004",
            "auth": "http://auth_service:8005"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def call_service(self, service_name: str, endpoint: str, 
                          method: str = "GET", **kwargs) -> Optional[Dict[str, Any]]:
        """Call another service"""
        try:
            base_url = self.services.get(service_name)
            if not base_url:
                raise ValueError(f"Unknown service: {service_name}")
            
            url = f"{base_url}{endpoint}"
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
            
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
