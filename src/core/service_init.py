"""
Service initialization using centralized dependency manager
"""
import os
import logging
from .dependency_manager import DependencyManager, ServiceConfig, dependency_manager

logger = logging.getLogger(__name__)

def initialize_service_dependencies():
    """Initialize global dependency manager for all services"""
    config = ServiceConfig(
        database_url=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@database_service:5432/sdg_pipeline"),
        weaviate_url=os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080"),
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        environment=os.environ.get("ENVIRONMENT", "development")
    )
    
    global dependency_manager
    dependency_manager = DependencyManager(config)
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, config.log_level))
    
    logger.info(f"Initialized dependency manager for environment: {config.environment}")
    return dependency_manager
