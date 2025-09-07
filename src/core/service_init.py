import os
import logging
from .dependency_manager import dependency_manager as _dm

logger = logging.getLogger(__name__)

def initialize_service_dependencies():
    return _dm
    # config = {
    #     "database_url": os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@database_service:5432/sdg_pipeline"),
    #     "weaviate_url": os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080"),
    #     "redis_url": os.environ.get("REDIS_URL", "redis://redis:6379"),
    #     "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    #     "environment": os.environ.get("ENVIRONMENT", "development"),
    # }

    # # Configure logging
    # logging.basicConfig(level=getattr(logging, config["log_level"], logging.INFO))
    # logger.info(f"Initialized dependency manager for environment: {config['environment']}")
    # # Return the already-created global instance
    # return dependency_manager
