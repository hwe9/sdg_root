# src/core/logging_config.py
"""
Centralized logging configuration for SDG project
This module ensures consistent logging across all services
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class SDGFormatter(logging.Formatter):
    """Custom formatter for SDG project"""

    def __init__(self, include_service: bool = True):
        super().__init__()
        self.include_service = include_service

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with SDG-specific format"""

        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.utcnow().isoformat()

        # Add service name if configured
        if self.include_service and not hasattr(record, 'service'):
            # Extract service from logger name (e.g., 'src.api.main' -> 'api')
            logger_parts = record.name.split('.')
            if len(logger_parts) >= 3 and logger_parts[0] == 'src':
                record.service = logger_parts[1]
            else:
                record.service = 'unknown'

        # Standard format
        if self.include_service:
            format_str = '%(timestamp)s - %(service)s - %(levelname)s - %(name)s - %(message)s'
        else:
            format_str = '%(timestamp)s - %(levelname)s - %(name)s - %(message)s'

        # Add extra fields for errors
        if record.levelno >= logging.ERROR:
            format_str += ' - %(pathname)s:%(lineno)d'

        # Add exception info if present
        if record.exc_info:
            format_str += '\n%(exc_text)s'

        self._fmt = format_str
        return super().format(record)

class SDGLogger:
    """Centralized logger for SDG services"""

    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.logger = logging.getLogger(f"src.{service_name}")
        self._configured = False

    def configure(
        self,
        level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_dir: str = "/var/log/sdg",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """Configure logging for this service"""

        if self._configured:
            return

        # Clear existing handlers
        self.logger.handlers.clear()

        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)

        # Don't propagate to root logger
        self.logger.propagate = False

        formatter = SDGFormatter(include_service=True)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            try:
                # Ensure log directory exists
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)

                log_file = log_path / f"{self.service_name}.log"

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            except Exception as e:
                # Fallback to console if file logging fails
                console_error = logging.StreamHandler(sys.stderr)
                console_error.setLevel(logging.ERROR)
                console_error.setFormatter(SDGFormatter(include_service=True))
                error_logger = logging.getLogger("sdg.logging")
                error_logger.addHandler(console_error)
                error_logger.error(f"Failed to configure file logging: {e}")

        self._configured = True

    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        if not self._configured:
            self.configure()
        return self.logger

    def log_service_start(self, version: str = "1.0.0", extra_info: Optional[Dict[str, Any]] = None):
        """Log service startup"""
        logger = self.get_logger()
        logger.info(f"🚀 Starting {self.service_name} service v{version}")
        if extra_info:
            logger.info(f"Service configuration: {extra_info}")

    def log_service_stop(self, reason: str = "normal"):
        """Log service shutdown"""
        logger = self.get_logger()
        logger.info(f"🛑 Stopping {self.service_name} service: {reason}")

    def log_error_with_context(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log error with additional context"""
        logger = self.get_logger()

        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            log_data["context"] = context
        if user_id:
            log_data["user_id"] = user_id
        if request_id:
            log_data["request_id"] = request_id

        logger.error(f"Error occurred: {log_data}", exc_info=True)

    def log_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics"""
        logger = self.get_logger()

        log_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success
        }

        if metadata:
            log_data.update(metadata)

        if success:
            logger.info(f"Performance: {log_data}")
        else:
            logger.warning(f"Performance (failed): {log_data}")

# Global logger instances for each service
loggers = {
    "api": SDGLogger("api"),
    "auth": SDGLogger("auth"),
    "data_processing": SDGLogger("data_processing"),
    "data_retrieval": SDGLogger("data_retrieval"),
    "vectorization": SDGLogger("vectorization"),
    "content_extraction": SDGLogger("content_extraction"),
    "core": SDGLogger("core"),
}

def get_logger(service_name: str) -> logging.Logger:
    """Get logger for a specific service"""
    if service_name not in loggers:
        loggers[service_name] = SDGLogger(service_name)

    logger = loggers[service_name]

    # Auto-configure if not configured
    if not logger._configured:
        # Get configuration from environment
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        log_dir = os.getenv("LOG_DIR", "/var/log/sdg")

        logger.configure(
            level=log_level,
            log_to_file=log_to_file,
            log_dir=log_dir
        )

    return logger.get_logger()

def configure_root_logger():
    """Configure the root logger to prevent duplicate logs"""
    root_logger = logging.getLogger()

    # Remove all existing handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger to WARNING to avoid noise from third-party libraries
    root_logger.setLevel(logging.WARNING)

    # Add a simple handler for critical errors
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    formatter = SDGFormatter(include_service=False)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Configure root logger on import
configure_root_logger()

# Convenience functions for common logging patterns
def log_api_request(method: str, endpoint: str, status_code: int, duration_ms: float, user_id: Optional[str] = None):
    """Log API request"""
    logger = get_logger("api")
    log_data = {
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "user_id": user_id or "anonymous"
    }

    if status_code >= 400:
        logger.warning(f"API request: {log_data}")
    else:
        logger.info(f"API request: {log_data}")

def log_database_operation(operation: str, table: str, duration_ms: float, success: bool = True, row_count: Optional[int] = None):
    """Log database operation"""
    logger = get_logger("core")
    log_data = {
        "operation": operation,
        "table": table,
        "duration_ms": duration_ms,
        "success": success,
        "row_count": row_count
    }

    if success:
        logger.debug(f"Database operation: {log_data}")
    else:
        logger.error(f"Database operation failed: {log_data}")

def log_external_api_call(service: str, endpoint: str, method: str, status_code: int, duration_ms: float):
    """Log external API call"""
    logger = get_logger("core")
    log_data = {
        "external_service": service,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_ms": duration_ms
    }

    if status_code >= 400:
        logger.warning(f"External API call: {log_data}")
    else:
        logger.info(f"External API call: {log_data}")
