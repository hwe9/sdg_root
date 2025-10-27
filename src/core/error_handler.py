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
    NETWORK_ERROR = "NETWORK_ERROR"

@dataclass
class ErrorResponse:
    """Standardized error response format"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    request_id: Optional[str] = None
    service: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class SDGErrorHandler:
    """Centralized error handling for SDG services"""
    
    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
    
    def handle_exception(
        self,
        exc: Exception,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Handle any exception and return standardized error response"""
        
        # Extract request ID if available
        request_id = None
        if request:
            request_id = getattr(request.state, 'request_id', None)
        
        # Determine error type and create appropriate response
        if isinstance(exc, HTTPException):
            return self._handle_http_exception(exc, request_id, context)
        elif isinstance(exc, ValueError):
            return self._handle_validation_error(exc, request_id, context)
        elif isinstance(exc, ConnectionError):
            return self._handle_connection_error(exc, request_id, context)
        elif isinstance(exc, TimeoutError):
            return self._handle_timeout_error(exc, request_id, context)
        elif isinstance(exc, FileNotFoundError):
            return self._handle_file_error(exc, request_id, context)
        else:
            return self._handle_generic_error(exc, request_id, context)
    
    def _handle_http_exception(
        self,
        exc: HTTPException,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        
        # Map HTTP status codes to error codes
        status_to_code = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.AUTHENTICATION_ERROR,
            403: ErrorCode.AUTHORIZATION_ERROR,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            429: ErrorCode.RATE_LIMIT_EXCEEDED,
            500: ErrorCode.INTERNAL_ERROR,
            502: ErrorCode.SERVICE_UNAVAILABLE,
            503: ErrorCode.SERVICE_UNAVAILABLE,
            504: ErrorCode.SERVICE_TIMEOUT,
        }
        
        error_code = status_to_code.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        
        error_response = ErrorResponse(
            error_code=error_code.value,
            message=str(exc.detail),
            request_id=request_id,
            service=self.service_name,
            details=context
        )
        
        self.logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.__dict__
        )
    
    def _handle_validation_error(
        self,
        exc: ValueError,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle validation errors"""
        
        error_response = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message=str(exc),
            request_id=request_id,
            service=self.service_name,
            details=context
        )
        
        self.logger.warning(f"Validation error: {exc}")
        
        return JSONResponse(
            status_code=400,
            content=error_response.__dict__
        )
    
    def _handle_connection_error(
        self,
        exc: ConnectionError,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle connection errors"""
        
        error_response = ErrorResponse(
            error_code=ErrorCode.DATABASE_CONNECTION_ERROR.value,
            message="Database connection failed",
            request_id=request_id,
            service=self.service_name,
            details={
                "original_error": str(exc),
                **(context or {})
            }
        )
        
        self.logger.error(f"Connection error: {exc}")
        
        return JSONResponse(
            status_code=503,
            content=error_response.__dict__
        )
    
    def _handle_timeout_error(
        self,
        exc: TimeoutError,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle timeout errors"""
        
        error_response = ErrorResponse(
            error_code=ErrorCode.SERVICE_TIMEOUT.value,
            message="Service request timed out",
            request_id=request_id,
            service=self.service_name,
            details={
                "original_error": str(exc),
                **(context or {})
            }
        )
        
        self.logger.error(f"Timeout error: {exc}")
        
        return JSONResponse(
            status_code=504,
            content=error_response.__dict__
        )
    
    def _handle_file_error(
        self,
        exc: FileNotFoundError,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle file-related errors"""
        
        error_response = ErrorResponse(
            error_code=ErrorCode.FILE_PROCESSING_ERROR.value,
            message="File not found or processing failed",
            request_id=request_id,
            service=self.service_name,
            details={
                "original_error": str(exc),
                **(context or {})
            }
        )
        
        self.logger.error(f"File error: {exc}")
        
        return JSONResponse(
            status_code=404,
            content=error_response.__dict__
        )
    
    def _handle_generic_error(
        self,
        exc: Exception,
        request_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> JSONResponse:
        """Handle generic errors"""
        
        error_response = ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR.value,
            message="An internal error occurred",
            request_id=request_id,
            service=self.service_name,
            details={
                "original_error": str(exc),
                "traceback": traceback.format_exc(),
                **(context or {})
            }
        )
        
        self.logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=error_response.__dict__
        )
    
    def create_error_response(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """Create a standardized error response"""
        
        error_response = ErrorResponse(
            error_code=error_code.value,
            message=message,
            details=details,
            request_id=request_id,
            service=self.service_name
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.__dict__
        )
    
    def log_error(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """Log an error with standardized format"""
        
        log_data = {
            "error_code": error_code.value,
            "message": message,
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            log_data["details"] = details
        
        if exc_info:
            self.logger.error(f"Error: {log_data}", exc_info=True)
        else:
            self.logger.error(f"Error: {log_data}")

# Global error handler instances for each service
error_handlers = {
    "api": SDGErrorHandler("api"),
    "auth": SDGErrorHandler("auth"),
    "data_processing": SDGErrorHandler("data_processing"),
    "data_retrieval": SDGErrorHandler("data_retrieval"),
    "vectorization": SDGErrorHandler("vectorization"),
    "content_extraction": SDGErrorHandler("content_extraction"),
}

def get_error_handler(service_name: str) -> SDGErrorHandler:
    """Get error handler for a specific service"""
    return error_handlers.get(service_name, SDGErrorHandler(service_name))

def create_error_middleware(service_name: str):
    """Create error handling middleware for FastAPI"""
    error_handler = get_error_handler(service_name)
    
    async def error_middleware(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return error_handler.handle_exception(exc, request)
    
    return error_middleware