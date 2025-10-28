# src/core/error_handler.py
"""
Centralized error handling for SDG project
This module ensures consistent error handling across all services
"""

import logging
import traceback
from typing import Dict
from typing import Any
from typing import Optional
from typing import Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
import inspect
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Standardized error codes"""
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Database errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_QUERY_ERROR = "DATABASE_QUERY_ERROR"
    DATABASE_TRANSACTION_ERROR = "DATABASE_TRANSACTION_ERROR"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    
    # Data processing errors
    DATA_VALIDATION_ERROR = "DATA_VALIDATION_ERROR"
    DATA_PROCESSING_ERROR = "DATA_PROCESSING_ERROR"
    FILE_PROCESSING_ERROR = "FILE_PROCESSING_ERROR"
    
    # External service errors
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

@dataclass
class ErrorResponse:
    """Standardized error response structure"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class CircuitBreaker:
    """Simple circuit breaker implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

    def _should_attempt_reset(self):
        if self.last_failure_time is None:
            return True
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout

class ErrorHandler:
    """Centralized error handler"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def handle_error(self, error: Exception, error_code: ErrorCode, 
                    message: str = None, details: Dict[str, Any] = None) -> ErrorResponse:
        """Handle and format errors consistently"""
        
        if message is None:
            message = str(error)
        
        error_response = ErrorResponse(
            error_code=error_code.value,
            message=message,
            details=details or {}
        )
        
        # Log the error
        logger.error(f"Error {error_code.value}: {message}", 
                    extra={"error_details": details, "traceback": traceback.format_exc()})
        
        return error_response
    
    def create_http_exception(self, error_code: ErrorCode, message: str, 
                            details: Dict[str, Any] = None, status_code: int = 500) -> HTTPException:
        """Create standardized HTTP exception"""
        error_response = self.handle_error(
            Exception(message), error_code, message, details
        )
        
        return HTTPException(
            status_code=status_code,
            detail=error_response.__dict__
        )

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
def handle_error(error: Exception, error_code: ErrorCode, 
                message: str = None, details: Dict[str, Any] = None) -> ErrorResponse:
    """Convenience function for error handling"""
    return error_handler.handle_error(error, error_code, message, details)

def create_http_exception(error_code: ErrorCode, message: str, 
                        details: Dict[str, Any] = None, status_code: int = 500) -> HTTPException:
    """Convenience function for creating HTTP exceptions"""
    return error_handler.create_http_exception(error_code, message, details, status_code)

def handle_errors(status_code: int = 500):
    """Decorator to standardize endpoint error handling.

    Wraps sync/async callables and converts exceptions to HTTPException
    with a consistent error response structure.
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    # Already standardized upstream
                    raise
                except SDGException as e:
                    raise create_http_exception(e.error_code, str(e), e.details, status_code=status_code)
                except Exception as e:
                    raise create_http_exception(ErrorCode.INTERNAL_ERROR, str(e), None, status_code=status_code)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except HTTPException:
                    raise
                except SDGException as e:
                    raise create_http_exception(e.error_code, str(e), e.details, status_code=status_code)
                except Exception as e:
                    raise create_http_exception(ErrorCode.INTERNAL_ERROR, str(e), None, status_code=status_code)
            return sync_wrapper
    return decorator

# Custom exception classes
class SDGException(Exception):
    """Base exception for SDG project"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INTERNAL_ERROR, 
                 details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ValidationError(SDGException):
    """Validation error"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)

class DatabaseError(SDGException):
    """Database error"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCode.DATABASE_CONNECTION_ERROR, details)

class ServiceError(SDGException):
    """Service error"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCode.SERVICE_UNAVAILABLE, details)

class ExternalServiceError(SDGException):
    """External service error"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCode.EXTERNAL_SERVICE_ERROR, details)

# Backwards compatibility alias
class SDGPipelineError(SDGException):
    pass

# Explicit exports
__all__ = [
    "ErrorCode",
    "ErrorResponse",
    "CircuitBreaker",
    "ErrorHandler",
    "error_handler",
    "handle_error",
    "create_http_exception",
    "handle_errors",
    "SDGException",
    "ValidationError",
    "DatabaseError",
    "ServiceError",
    "ExternalServiceError",
    "SDGPipelineError",
]