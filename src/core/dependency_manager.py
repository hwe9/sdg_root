# /sdg_root/src/core/dependency_manager.py
"""
Comprehensive Dependency Manager for SDG Pipeline
Handles service dependencies, health checks, and startup orchestration
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import httpx
from datetime import datetime, timedelta
import threading
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    PENDING = "pending"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class ServiceDependency:
    """Represents a service dependency"""
    name: str
    url: str
    health_endpoint: str = "/health"
    timeout: int = 30
    required: bool = True
    retry_attempts: int = 3
    retry_delay: float = 2.0
    dependencies: List[str] = field(default_factory=list)
    
class DependencyManager:
    """
    Manages service dependencies and startup orchestration
    Integrates with your existing service architecture
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.services: Dict[str, ServiceDependency] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.startup_tasks: Dict[str, Callable] = {}
        self.cleanup_tasks: Dict[str, Callable] = {}
        self.health_check_interval = self.config.get("health_check_interval", 60)
        self.max_startup_time = self.config.get("max_startup_time", 300)  # 5 minutes
        self._monitoring_task: Optional[asyncio.Task] = None
        self._startup_complete = asyncio.Event()
        
        # Initialize core SDG services
        self._register_core_services()
    
    def _register_core_services(self):
        """Register core SDG pipeline services"""
        core_services = {
            "database": ServiceDependency(
                name="database",
                url="postgresql://localhost:5432",
                health_endpoint="/",  # Custom health check
                required=True,
                dependencies=[]
            ),
            "weaviate": ServiceDependency(
                name="weaviate",
                url="http://weaviate_service:8080",
                health_endpoint="/v1/meta",
                required=True,
                dependencies=["weaviate_transformer"]
            ),
            "weaviate_transformer": ServiceDependency(
                name="weaviate_transformer",
                url="http://weaviate_transformer_service:8080",
                health_endpoint="/.well-known/ready",
                required=True,
                timeout=60,
                dependencies=[]
            ),
            "auth": ServiceDependency(
                name="auth",
                url="http://auth_service:8005",
                required=True,
                dependencies=["database"]
            ),
            "api": ServiceDependency(
                name="api",
                url="http://api_service:8000",
                required=True,
                dependencies=["database", "auth"]
            ),
            "data_retrieval": ServiceDependency(
                name="data_retrieval",
                url="http://data_retrieval_service:8002",
                required=True,
                dependencies=["database"]
            ),
            "data_processing": ServiceDependency(
                name="data_processing",
                url="http://data_processing_service:8001",
                required=True,
                dependencies=["database", "weaviate", "data_retrieval"]
            ),
            "vectorization": ServiceDependency(
                name="vectorization",
                url="http://vectorization_service:8003",
                required=True,
                dependencies=["weaviate", "database"]
            ),
            "content_extraction": ServiceDependency(
                name="content_extraction",
                url="http://content_extraction_service:8004",
                required=False,  # Optional service
                dependencies=["database"]
            )
        }
        
        for service_name, service_dep in core_services.items():
            self.register_service(service_dep)
    
    def register_service(self, service: ServiceDependency):
        """Register a service dependency"""
        self.services[service.name] = service
        self.service_status[service.name] = ServiceStatus.PENDING
        logger.info(f"Registered service: {service.name}")
    
    def register_startup_task(self, service_name: str, task: Callable):
        """Register a startup task for a service"""
        self.startup_tasks[service_name] = task
        logger.info(f"Registered startup task for: {service_name}")
    
    def register_cleanup_task(self, service_name: str, task: Callable):
        """Register a cleanup task for a service"""
        self.cleanup_tasks[service_name] = task
        logger.info(f"Registered cleanup task for: {service_name}")
    
    async def start_all_services(self) -> Dict[str, ServiceStatus]:
        """Start all services in dependency order"""
        logger.info("🚀 Starting SDG Pipeline services...")
        
        # Calculate startup order based on dependencies
        startup_order = self._calculate_startup_order()
        logger.info(f"Service startup order: {startup_order}")
        
        # Start services in order
        for service_name in startup_order:
            await self._start_service(service_name)
        
        # Start health monitoring
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_services())
        
        # Signal startup complete
        self._startup_complete.set()
        
        logger.info("✅ All services started successfully!")
        return self.service_status.copy()
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate service startup order using topological sort"""
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for service_name, service in self.services.items():
            for dep in service.dependencies:
                if dep in self.services:
                    graph[dep].append(service_name)
                    in_degree[service_name] += 1
            if service_name not in in_degree:
                in_degree[service_name] = 0
        
        # Topological sort
        queue = [service for service, degree in in_degree.items() if degree == 0]
        startup_order = []
        
        while queue:
            service = queue.pop(0)
            startup_order.append(service)
            
            for dependent in graph[service]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(startup_order) != len(self.services):
            missing = set(self.services.keys()) - set(startup_order)
            logger.error(f"Circular dependency detected. Missing services: {missing}")
            raise ValueError(f"Circular dependency in services: {missing}")
        
        return startup_order
    
    async def _start_service(self, service_name: str):
        """Start an individual service"""
        service = self.services[service_name]
        logger.info(f"🔄 Starting service: {service_name}")
        
        self.service_status[service_name] = ServiceStatus.STARTING
        
        try:
            # Wait for dependencies
            await self._wait_for_dependencies(service)
            
            # Run startup task if available
            if service_name in self.startup_tasks:
                await self._run_startup_task(service_name)
            
            # Health check
            if await self._health_check_service(service):
                self.service_status[service_name] = ServiceStatus.HEALTHY
                logger.info(f"✅ Service started successfully: {service_name}")
            else:
                raise Exception(f"Health check failed for {service_name}")
                
        except Exception as e:
            self.service_status[service_name] = ServiceStatus.FAILED
            logger.error(f"❌ Failed to start service {service_name}: {e}")
            
            if service.required:
                raise Exception(f"Required service {service_name} failed to start: {e}")
    
    async def _wait_for_dependencies(self, service: ServiceDependency):
        """Wait for service dependencies to be healthy"""
        if not service.dependencies:
            return
        
        logger.info(f"Waiting for dependencies of {service.name}: {service.dependencies}")
        
        start_time = time.time()
        while time.time() - start_time < self.max_startup_time:
            all_ready = True
            
            for dep_name in service.dependencies:
                if dep_name not in self.service_status:
                    continue
                    
                if self.service_status[dep_name] != ServiceStatus.HEALTHY:
                    all_ready = False
                    break
            
            if all_ready:
                logger.info(f"All dependencies ready for {service.name}")
                return
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"Dependencies not ready for {service.name} within {self.max_startup_time}s")
    
    async def _run_startup_task(self, service_name: str):
        """Run startup task for service"""
        task = self.startup_tasks[service_name]
        logger.info(f"Running startup task for {service_name}")
        
        try:
            if asyncio.iscoroutinefunction(task):
                await task()
            else:
                await asyncio.get_event_loop().run_in_executor(None, task)
        except Exception as e:
            logger.error(f"Startup task failed for {service_name}: {e}")
            raise
    
    async def _health_check_service(self, service: ServiceDependency) -> bool:
        """Perform health check on service"""
        if service.name == "database":
            return await self._check_database_health()
        
        for attempt in range(service.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=service.timeout) as client:
                    url = f"{service.url}{service.health_endpoint}"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        status = health_data.get("status", "unknown")
                        return status in ["healthy", "ok", "ready"]
                    
            except Exception as e:
                logger.warning(f"Health check attempt {attempt + 1} failed for {service.name}: {e}")
                if attempt < service.retry_attempts - 1:
                    await asyncio.sleep(service.retry_delay)
        
        return False
    
    async def _check_database_health(self) -> bool:
        """Custom database health check"""
        try:
            from .db_utils import check_database_health
            return check_database_health()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def _monitor_services(self):
        """Continuously monitor service health"""
        logger.info("🔍 Starting service health monitoring...")
        
        while True:
            try:
                for service_name, service in self.services.items():
                    if self.service_status[service_name] == ServiceStatus.HEALTHY:
                        is_healthy = await self._health_check_service(service)
                        
                        if not is_healthy:
                            logger.warning(f"⚠️ Service {service_name} became unhealthy")
                            self.service_status[service_name] = ServiceStatus.UNHEALTHY
                            
                            # Attempt recovery for critical services
                            if service.required:
                                await self._attempt_service_recovery(service_name)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("Service monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _attempt_service_recovery(self, service_name: str):
        """Attempt to recover a failed service"""
        logger.info(f"🔧 Attempting recovery for service: {service_name}")
        
        max_recovery_attempts = 3
        for attempt in range(max_recovery_attempts):
            try:
                # Wait a bit before retry
                await asyncio.sleep(5 * (attempt + 1))
                
                # Try to restart the service
                await self._start_service(service_name)
                
                if self.service_status[service_name] == ServiceStatus.HEALTHY:
                    logger.info(f"✅ Service {service_name} recovered successfully")
                    return
                    
            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed for {service_name}: {e}")
        
        logger.error(f"❌ Failed to recover service {service_name} after {max_recovery_attempts} attempts")
    
    async def shutdown_all_services(self):
        """Shutdown all services gracefully"""
        logger.info("🔄 Shutting down SDG Pipeline services...")
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown in reverse dependency order
        startup_order = self._calculate_startup_order()
        shutdown_order = list(reversed(startup_order))
        
        for service_name in shutdown_order:
            await self._shutdown_service(service_name)
        
        logger.info("✅ All services shut down successfully")
    
    async def _shutdown_service(self, service_name: str):
        """Shutdown individual service"""
        logger.info(f"🔄 Shutting down service: {service_name}")
        
        try:
            # Run cleanup task if available
            if service_name in self.cleanup_tasks:
                cleanup_task = self.cleanup_tasks[service_name]
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    await asyncio.get_event_loop().run_in_executor(None, cleanup_task)
            
            self.service_status[service_name] = ServiceStatus.STOPPED
            logger.info(f"✅ Service shut down: {service_name}")
            
        except Exception as e:
            logger.error(f"Error shutting down service {service_name}: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current status of all services"""
        return {
            "services": {
                name: {
                    "status": status.value,
                    "required": self.services[name].required,
                    "dependencies": self.services[name].dependencies,
                    "health_url": f"{self.services[name].url}{self.services[name].health_endpoint}"
                }
                for name, status in self.service_status.items()
            },
            "overall_status": self._get_overall_status(),
            "startup_complete": self._startup_complete.is_set(),
            "last_check": datetime.utcnow().isoformat()
        }
    
    def _get_overall_status(self) -> str:
        """Get overall system status"""
        required_services = [
            name for name, service in self.services.items() 
            if service.required
        ]
        
        required_statuses = [
            self.service_status[name] for name in required_services
        ]
        
        if all(status == ServiceStatus.HEALTHY for status in required_statuses):
            return "healthy"
        elif any(status == ServiceStatus.FAILED for status in required_statuses):
            return "critical"
        elif any(status == ServiceStatus.UNHEALTHY for status in required_statuses):
            return "degraded"
        else:
            return "starting"
    
    @asynccontextmanager
    async def managed_startup(self):
        """Context manager for complete service lifecycle"""
        try:
            await self.start_all_services()
            yield self
        finally:
            await self.shutdown_all_services()

# Global dependency manager instance
dependency_manager = DependencyManager()

# Convenience functions for FastAPI integration
async def wait_for_dependencies(*service_names: str):
    """Wait for specific services to be healthy"""
    await dependency_manager._startup_complete.wait()
    
    for service_name in service_names:
        if service_name in dependency_manager.service_status:
            status = dependency_manager.service_status[service_name]
            if status != ServiceStatus.HEALTHY:
                raise Exception(f"Required service {service_name} is not healthy: {status.value}")

async def get_dependency_status():
    """Get dependency status for health endpoints"""
    return dependency_manager.get_service_status()

# Integration with your existing services
def setup_sdg_dependencies():
    """Setup SDG-specific dependency configurations"""
    
    # Register AI model startup tasks
    async def initialize_ai_models():
        """Initialize Whisper and Sentence Transformer models"""
        logger.info("Initializing AI models for data processing...")
        # Your existing model initialization logic here
        
    dependency_manager.register_startup_task("data_processing", initialize_ai_models)
    
    # Register database initialization
    async def initialize_database():
        """Initialize database with SDG schema"""
        logger.info("Initializing SDG database schema...")
        try:
            from .db_init import initialize_database
            initialize_database()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    dependency_manager.register_startup_task("database", initialize_database)
    
    # Register vector database setup
    async def setup_weaviate_schema():
        """Setup Weaviate schema for SDG data"""
        logger.info("Setting up Weaviate schema...")
        # Your existing schema setup logic here
        
    dependency_manager.register_startup_task("weaviate", setup_weaviate_schema)

# Usage example for your FastAPI applications
def create_app_with_dependencies():
    """Example of integrating with FastAPI"""
    from fastapi import FastAPI, Depends
    
    app = FastAPI()
    
    @app.on_event("startup")
    async def startup_event():
        setup_sdg_dependencies()
        await dependency_manager.start_all_services()
    
    @app.on_event("shutdown") 
    async def shutdown_event():
        await dependency_manager.shutdown_all_services()
    
    @app.get("/health")
    async def health_check():
        return await get_dependency_status()
    
    # Dependency injection for endpoints
    @app.get("/api/data")
    async def get_data(deps=Depends(lambda: wait_for_dependencies("database", "vectorization"))):
        # Your endpoint logic here
        return {"status": "ok"}
    
    return app
