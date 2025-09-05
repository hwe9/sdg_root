import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from ..core.secrets_manager import secrets_manager
from ..core.dependency_manager import get_dependency_manager
import logging

logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL with encrypted credentials"""
    try:
        db_host = os.environ.get("DB_HOST", "database_service")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = secrets_manager.get_secret("POSTGRES_DB")
        db_user = secrets_manager.get_secret("POSTGRES_USER")
        db_password = secrets_manager.get_secret("POSTGRES_PASSWORD")

        if not all([db_user, db_password, db_name]):
            logger.error("Required database credentials missing")
            raise ValueError("Database configuration incomplete")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    except Exception as e:
        logger.error(f"Error constructing database URL: {e}")
        raise

DATABASE_URL = get_database_url()

# Add connection security settings
engine_kwargs = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_size": 10,
    "max_overflow": 20,
    "connect_args": {
        "options": "-c default_transaction_isolation=serializable"
    }
}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs.update({
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False}
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get database session via dependency manager"""
    dep_manager = get_dependency_manager()
    return dep_manager.get_database_session()

def check_database_health():
    """Check if database is accessible"""
    try:
        dep_manager = get_dependency_manager()
        with dep_manager.get_database_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
