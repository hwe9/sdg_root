# src/core/db_utils.py
import os
import logging
from urllib.parse import quote_plus
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import DisconnectionError

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration management"""
    
    def __init__(self):
        self._url = None
        self._engine = None
    
    def get_database_url(self) -> str:
        """Get database URL with consistent fallback logic"""
        if self._url:
            return self._url
            
        # Try environment variable first
        explicit_url = os.environ.get("DATABASE_URL")
        if explicit_url:
            self._url = explicit_url
            return self._url
        
        # Construct from individual components with consistent defaults
        user = os.getenv("POSTGRES_USER", "postgres")
        password = quote_plus(os.getenv("POSTGRES_PASSWORD", "postgres"))
        host = os.getenv("DB_HOST", "database_service")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("POSTGRES_DB", "sdg_pipeline")
        
        self._url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        return self._url
    
    def get_engine(self) -> Engine:
        """Get database engine with consistent configuration"""
        if self._engine:
            return self._engine
            
        engine_kwargs = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 10,
            "max_overflow": 20,
            "echo": False,
            "poolclass": QueuePool,
            "connect_args": {
                "connect_timeout": 30,
                "application_name": "SDG_Pipeline"
            }
        }
        
        self._engine = create_engine(self.get_database_url(), **engine_kwargs)
        return self._engine
    
    def check_database_health(self) -> bool:
        """Consistent database health check across all services"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except (OperationalError, DisconnectionError) as e:
            logger.error(f"Database health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected database health check error: {e}")
            return False

# Global database configuration instance
db_config = DatabaseConfig()

# Convenience functions for backward compatibility
def get_database_url() -> str:
    return db_config.get_database_url()

def get_database_engine() -> Engine:
    return db_config.get_engine()

def check_database_health() -> bool:
    return db_config.check_database_health()