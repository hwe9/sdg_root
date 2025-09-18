# src/core/db_utils.py
import logging
logger = logging.getLogger(__name__)
try:
    from src.data_processing.core import db_utils as _impl  # noqa: F401
except Exception as e:
    _impl = None
    logger.error(f"Failed to import src.data_processing.core.db_utils: {e}")

def check_database_health(*args, **kwargs) -> bool:
    fn = getattr(_impl, "check_database_health", None)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    logger.error("check_database_health not found in data_processing.core.db_utils")
    return False
