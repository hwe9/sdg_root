"""
Enhanced SDG Data Retrieval Service
Production-ready FastAPI service with dependency management integration
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .core.retrieval_engine import RetrievalEngine
from .core.source_manager import SourceManager
from ..core.dependency_manager import dependency_manager, wait_for_dependencies, get_dependency_status
from ..core.error_handler import handle_errors, SDGPipelineError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enhanced Request/Response Models
class RetrievalRequest(BaseModel):
    sources: Optional[list] = Field(None, description="Specific sources to retrieve from")
    force_refresh: bool = Field(False, description="Force refresh of already processed sources")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent retrieval processes")
    filter_region: Optional[str] = Field(None, description="Filter sources by region")
    sdg_goals: Optional[list[int]] = Field(None, description="Filter for specific SDG goals")


class RetrievalStatus(BaseModel):
    status: str
    message: str
    started_at: Optional[datetime] = None
    processed_count: int = 0
    failed_count: int = 0
    current_source: Optional[str] = None
    estimated_completion: Optional[datetime] = None


# Global components
retrieval_engine: Optional[RetrievalEngine] = None
source_manager: Optional[SourceManager] = None


async def setup_data_retrieval_dependencies():
    """Setup data retrieval service dependencies"""
    logger.info("Setting up data retrieval service dependencies...")
    
    # Register this service with dependency manager
    from ..core.dependency_manager import ServiceDependency
    
    data_retrieval_service = ServiceDependency(
        name="data_retrieval",
        url=f"http://localhost:{os.getenv('SERVICE_PORT', 8002)}",
        health_endpoint="/health",
        required=True,
        dependencies=["database"],
        timeout=30
    )
    
    dependency_manager.register_service(data_retrieval_service)
    
    # Register startup task for this service
    async def initialize_retrieval_components():
        """Initialize retrieval engine and source manager"""
        global retrieval_engine, source_manager
        
        logger.info("Initializing retrieval components...")
        
        try:
            # Initialize source manager
            source_manager = SourceManager(
                sources_file=os.getenv("SOURCES_FILE", "/app/quelle.txt"),
                data_dir=os.getenv("DATA_DIR", "/app/raw_data")
            )
            
            # Initialize retrieval engine
            retrieval_engine = RetrievalEngine(
                source_manager=source_manager,
                data_dir=os.getenv("DATA_DIR", "/app/raw_data"),
                processed_file=os.getenv("PROCESSED_FILE", "/app/processed_data/processed_data.json")
            )
            
            await retrieval_engine.initialize()
            logger.info("✅ Retrieval components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval components: {e}")
            raise SDGPipelineError(f"Retrieval initialization failed: {e}")
    
    dependency_manager.register_startup_task("data_retrieval", initialize_retrieval_components)
    
    # Register cleanup task
    async def cleanup_retrieval_components():
        """Cleanup retrieval components"""
        global retrieval_engine, source_manager
        
        logger.info("Cleaning up retrieval components...")
        
        if retrieval_engine:
            await retrieval_engine.cleanup()
        if source_manager:
            await source_manager.cleanup()
            
        logger.info("✅ Retrieval components cleaned up")
    
    dependency_manager.register_cleanup_task("data_retrieval", cleanup_retrieval_components)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with dependency management"""
    global retrieval_engine, source_manager
    
    # Startup
    logger.info("🚀 Starting Enhanced Data Retrieval Service...")
    
    try:
        # Setup dependencies
        await setup_data_retrieval_dependencies()
        
        # Wait for required dependencies (database)
        await wait_for_dependencies("database")
        
        # Initialize components through dependency manager
        if "data_retrieval" in dependency_manager.startup_tasks:
            await dependency_manager.startup_tasks["data_retrieval"]()
        
        logger.info("✅ Enhanced data retrieval service initialized with dependency management")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize retrieval service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Data Retrieval Service...")
    
    try:
        if "data_retrieval" in dependency_manager.cleanup_tasks:
            await dependency_manager.cleanup_tasks["data_retrieval"]()
        
        logger.info("✅ Enhanced data retrieval service shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SDG Data Retrieval Service",
    description="Enhanced service for retrieving SDG-related content with dependency management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Dependency injection
async def get_retrieval_engine() -> RetrievalEngine:
    """Get retrieval engine with dependency check"""
    await wait_for_dependencies("data_retrieval")
    if not retrieval_engine:
        raise HTTPException(status_code=503, detail="Retrieval engine not initialized")
    return retrieval_engine


async def get_source_manager() -> SourceManager:
    """Get source manager with dependency check"""
    await wait_for_dependencies("data_retrieval")
    if not source_manager:
        raise HTTPException(status_code=503, detail="Source manager not initialized")
    return source_manager


# API Endpoints
@app.get("/health")
async def health_check():
    """Comprehensive service health check with dependency status"""
    try:
        # Get dependency status
        dependency_status = await get_dependency_status()
        
        # Check local components
        engine_health = retrieval_engine.health_check() if retrieval_engine else {"status": "not_initialized"}
        source_health = source_manager.health_check() if source_manager else {"status": "not_initialized"}
        
        # Determine overall status
        overall_status = "healthy"
        
        if engine_health.get("status") != "healthy" or source_health.get("status") != "healthy":
            overall_status = "unhealthy"
        
        if dependency_status.get("overall_status") not in ["healthy", "starting"]:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "service": "SDG Data Retrieval Service",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "retrieval_engine": engine_health,
                "source_manager": source_health
            },
            "dependencies": dependency_status,
            "startup_complete": dependency_manager._startup_complete.is_set()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/retrieve", response_model=RetrievalStatus)
@handle_errors()
async def start_retrieval(
    request: RetrievalRequest,
    background_tasks: BackgroundTasks,
    engine: RetrievalEngine = Depends(get_retrieval_engine)
):
    """Start enhanced data retrieval process with dependency validation"""
    try:
        # Validate request parameters
        if request.max_concurrent > 20:
            raise HTTPException(status_code=400, detail="max_concurrent cannot exceed 20")
        
        # Start retrieval process in background
        background_tasks.add_task(
            engine.run_retrieval,
            sources=request.sources,
            force_refresh=request.force_refresh,
            max_concurrent=request.max_concurrent,
            filter_region=request.filter_region,
            sdg_goals=request.sdg_goals
        )
        
        logger.info(f"Started retrieval process with {request.max_concurrent} concurrent workers")
        
        return RetrievalStatus(
            status="started",
            message="Enhanced retrieval process initiated with dependency validation",
            started_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error starting retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
@handle_errors()
async def get_sources(manager: SourceManager = Depends(get_source_manager)):
    """Get configured data sources with validation status"""
    try:
        summary = manager.get_source_summary()
        
        # Add dependency information
        summary["dependency_status"] = await get_dependency_status()
        summary["last_updated"] = datetime.utcnow().isoformat()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
@handle_errors() 
async def get_retrieval_status(engine: RetrievalEngine = Depends(get_retrieval_engine)):
    """Get current retrieval status with detailed metrics"""
    try:
        status = engine.get_status()
        
        # Add system information
        status["system_info"] = {
            "dependencies_healthy": dependency_manager._get_overall_status() == "healthy",
            "service_uptime": datetime.utcnow().isoformat(),
            "active_workers": engine.get_active_worker_count() if hasattr(engine, 'get_active_worker_count') else 0
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dependencies")
async def get_service_dependencies():
    """Get current dependency status for monitoring"""
    return await get_dependency_status()


@app.post("/validate-sources")
@handle_errors()
async def validate_sources(manager: SourceManager = Depends(get_source_manager)):
    """Validate all configured sources for accessibility"""
    try:
        validation_results = await manager.validate_all_sources()
        return {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "results": validation_results
        }
        
    except Exception as e:
        logger.error(f"Error validating sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", 8002))
    reload = os.getenv("ENVIRONMENT", "production") != "production"
    
    logger.info(f"Starting Data Retrieval Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
