# /sdg_root/src/core/dependency_manager.py
import os, random
import asyncio
import logging
import json, time
from fastapi import FastAPI, Depends
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import httpx
from datetime import datetime
from collections import defaultdict
from prometheus_client import Counter, Gauge, Histogram
from .error_handler import CircuitBreaker

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
    name: str
    url: str
    health_endpoint: str = "/health"
    timeout: int = 30
    required: bool = True
    retry_attempts: int = 3
    retry_delay: float = 2.0
    dependencies: List[str] = field(default_factory=list)


class DependencyManager:
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
        self._db_engine = None
        self._SessionLocal = None
        self._db_lock = asyncio.Lock()
        self._vector_lock = asyncio.Lock()
        self._vector_url = None
        self._vector_api_key = None
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._state: Dict[str, Dict[str, int]] = {}  # {"service": {"fails": int, "oks": int}}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._hc_duration = Histogram("health_check_duration_seconds", "Duration of dependency health checks", ["service"])
        self._hc_total = Counter("health_checks_total", "Total dependency health checks", ["service", "result"])
        self._svc_status = Gauge("service_status", "0=down,1=up", ["service"])
        self._register_core_services()

    def _log_event(self, event: str, service: str, **kw):
        payload = {"event": event, "service": service, **kw}
        logging.getLogger(__name__).info(json.dumps(payload))


    def _backoff_delay(self, attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
        exp = min(cap, base * (2 ** attempt))
        return random.uniform(0, exp)

    def _get_breaker(self, service: ServiceDependency) -> CircuitBreaker:
        if service.name not in self._breakers:
            self._breakers[service.name] = CircuitBreaker(
                failure_threshold=getattr(service, "fail_max", 5),
                recovery_timeout=getattr(service, "reset_timeout", 60),
            )
        return self._breakers[service.name]

    def _update_status_with_hysteresis(
        self,
        service_name: str,
        probe_ok: bool,
        ok_threshold: int = 2,
        fail_threshold: int = 3,
    ):
        st = self._state.setdefault(service_name, {"fails": 0, "oks": 0})
        if probe_ok:
            st["oks"] += 1
            st["fails"] = 0
            if st["oks"] >= ok_threshold:
                self.service_status[service_name] = ServiceStatus.HEALTHY
        else:
            st["fails"] += 1
            st["oks"] = 0
            if st["fails"] >= fail_threshold:
                # Only downgrade if not already failed on startup
                if self.service_status.get(service_name) != ServiceStatus.FAILED:
                    self.service_status[service_name] = ServiceStatus.UNHEALTHY

    
    async def _ensure_db(self):
        """Lazily create SQLAlchemy engine and session factory once."""
        if self._db_engine is not None and self._SessionLocal is not None:
            return

        async with self._db_lock:
            if self._db_engine is not None and self._SessionLocal is not None:
                return

            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                try:
                    from .config_manager import ConfigurationManager
                    db_url = ConfigurationManager().get_database_url()
                except Exception:
                    db_url = "postgresql://postgres:postgres@database_service:5432/sdg_pipeline"

            engine_kwargs = {
                "pool_pre_ping": True,
                "pool_recycle": 300,
                "pool_size": 10,
                "max_overflow": 20,
            }
            self._db_engine = create_engine(db_url, **engine_kwargs)
            self._SessionLocal = sessionmaker(bind=self._db_engine, autocommit=False, autoflush=False)
            logger.info("Database engine and session factory initialized")

    @asynccontextmanager
    async def get_db_connection(self):
        await self._ensure_db()
        try:
            # open transaction-bound connection
            with self._db_engine.begin() as connection:
                yield connection
        except Exception as e:
            logger.error(f"DB connection error: {e}")
            raise
    
    def get_database_session(self):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ensure_db(), loop)
                fut.result()
            except Exception:
                asyncio.run(self._ensure_db())
        else:
            asyncio.run(self._ensure_db())
        return self._SessionLocal()
    class _VectorClientAdapter:
        
        def __init__(self, weaviate_client):
            self._client = weaviate_client

        def insert_data(self, data: Dict[str, Any], class_name: str, vector: List[float]):
            # Ensure required fields exist in payload
            if not isinstance(data, dict):
                raise ValueError("data must be a dict")
            if not isinstance(vector, (list, tuple)):
                raise ValueError("vector must be a list/tuple of floats")

            # Create object with explicit vector
            self._client.data_object.create(
                data_object=data,
                class_name=class_name,
                vector=vector
            )

        def health_check(self) -> Dict[str, Any]:
            try:
                ready = self._client.is_ready()
                return {"status": "healthy" if ready else "unhealthy"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def close(self):
            try:
                # The Python client does not require explicit close,
                # but keep for symmetry/future change
                pass
            except Exception:
                pass

    @asynccontextmanager
    async def get_vector_client(self):
        """
        Async context manager that yields a _VectorClientAdapter instance.
        This matches usage in data_processing/core/db_utils:
            async with dep_manager.get_vector_client() as vector_client:
                vector_client.insert_data(...)
        """
        # Resolve URL/API key lazily
        async with self._vector_lock:
            if not self._vector_url:
                self._vector_url = os.getenv("WEAVIATE_URL", "http://weaviate_service:8080")
            if self._vector_api_key is None:
                # empty is allowed (anonymous disabled/enabled by config)
                self._vector_api_key = os.getenv("WEAVIATE_API_KEY", "")

        import weaviate

        # Build client (remote)
        auth_config = None
        if self._vector_api_key:
            try:
                auth_config = weaviate.AuthApiKey(api_key=self._vector_api_key)
            except Exception:
                auth_config = None

        client = weaviate.Client(
            url=self._vector_url,
            auth_client_secret=auth_config
        )

        adapter = self._VectorClientAdapter(client)
        try:
            yield adapter
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    # ------------------------------
    # Registration and orchestration
    # ------------------------------
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
                url=os.getenv("WEAVIATE_URL", "http://weaviate_service:8080"),
                health_endpoint="/v1/.well-known/ready",
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
        """Start all services by dependency levels (parallel per level)"""
        logger.info("🚀 Starting SDG Pipeline services...")
        levels = self._levels_from_toposort()
        logger.info(f"Service startup levels: {levels}")
        for lvl in levels:
            await asyncio.gather(*[self._start_service(s) for s in lvl], return_exceptions=False)
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_services())
        self._startup_complete.set()
        logger.info("✅ All services started successfully!")
        return self.service_status.copy()
    
    def _levels_from_toposort(self) -> List[List[str]]:
        """Group services by topological levels for parallel startup"""
        graph = defaultdict(list)
        indeg = defaultdict(int)
        for name, svc in self.services.items():
            for dep in svc.dependencies:
                if dep in self.services:
                    graph[dep].append(name)
                    indeg[name] += 1
            if name not in indeg:
                indeg[name] = 0
        # Kahn's algorithm by levels
        level = [s for s, d in indeg.items() if d == 0]
        levels = []
        visited = set()
        while level:
            levels.append(level[:])
            next_level = []
            for u in level:
                visited.add(u)
                for v in graph[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        next_level.append(v)
            level = next_level
        if len(visited) != len(self.services):
            missing = set(self.services.keys()) - visited
            logger.error(f"Circular dependency detected. Missing services: {missing}")
            raise ValueError(f"Circular dependency in services: {missing}")
        return levels

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
        if service.name == "database":
            for attempt in range(service.retry_attempts):
                try:
                    ok = await self._check_database_health()
                    if ok:
                        return True
                except Exception as e:
                    logger.warning(f"Database health check (attempt {attempt+1}) failed: {e}")
                await asyncio.sleep(self._backoff_delay(attempt))
            return False

        breaker = self._get_breaker(service)

        def _probe_sync() -> bool:
            url = f"{service.url}{service.health_endpoint}"
            resp = httpx.get(url, timeout=service.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("status") in ["healthy", "ok", "ready"]

        loop = asyncio.get_running_loop()
        for attempt in range(service.retry_attempts):
            try:
                ok = await loop.run_in_executor(None, lambda: breaker.call(_probe_sync))
                if ok:
                    return True
            except Exception as e:
                logger.warning(f"Health check (attempt {attempt+1}) failed for {service.name}: {e}")
            await asyncio.sleep(self._backoff_delay(attempt))
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
        logger.info("🔍 Starting service health monitoring...")
        while True:
            try:
                tasks = [
                    self._check_and_update(name, svc)
                    for name, svc in self.services.items()
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                logger.info("Service monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(10)

    async def _check_and_update(self, name: str, service: ServiceDependency):
        start = time.time()
        ok = await self._health_check_service(service)
        duration = time.time() - start
        # Prometheus metrics
        try:
            self._hc_duration.labels(service=name).observe(duration)
            self._hc_total.labels(service=name, result="ok" if ok else "fail").inc()
            self._svc_status.labels(service=name).set(1.0 if ok else 0.0)
        except Exception:
            pass
        # Hysteresis update
        self._update_status_with_hysteresis(name, ok)
        # Recovery only if required and transitioned to unhealthy
        if not ok and service.required and self.service_status.get(name) == ServiceStatus.UNHEALTHY:
            await self._attempt_service_recovery(name)
      

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

        try:
            if self._http_client:
                await self._http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")

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


def get_dependency_manager() -> DependencyManager:
    """Expose the global dependency manager for services that import it"""
    return dependency_manager


# Integration with your existing services
def setup_sdg_dependencies():
    """Setup SDG-specific dependency configurations"""

    # Register AI model startup tasks
    async def initialize_ai_models():
        """Initialize Whisper and Sentence Transformer models"""
        logger.info("Initializing AI models for data processing...")
        # Hook for model loading

    dependency_manager.register_startup_task("data_processing", initialize_ai_models)

    # Register database initialization
    async def initialize_database():
        """Initialize database with SDG schema"""
        logger.info("Initializing SDG database schema...")
        try:
            from .db_init import initialize_database as _init_db
            _init_db()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    dependency_manager.register_startup_task("database", initialize_database)

    # Register vector database setup
    async def setup_weaviate_schema():
        """Setup Weaviate schema for SDG data"""
        logger.info("Setting up Weaviate schema...")
        # Hook for schema setup

    dependency_manager.register_startup_task("weaviate", setup_weaviate_schema)


def create_app_with_dependencies():
    """Example of integrating with FastAPI using lifespan instead of on_event"""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Setup SDG-specific startup tasks
        setup_sdg_dependencies()
        # Start all services in dependency order
        await dependency_manager.start_all_services()
        try:
            yield
        finally:
            # Graceful shutdown in reverse dependency order
            await dependency_manager.shutdown_all_services()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return await get_dependency_status()

    # Dependency injection for endpoints
    @app.get("/api/data")
    async def get_data(deps=Depends(lambda: wait_for_dependencies("database", "vectorization"))):
        # Your endpoint logic here
        return {"status": "ok"}

    return app