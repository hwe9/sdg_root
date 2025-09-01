import logging
import traceback
import sys
from typing import Any, Dict, Optional, Type
from functools import wraps
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class SDGPipelineError(Exception):
    """Base exception for SDG Pipeline"""
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

class DatabaseError(SDGPipelineError):
    """Database operation errors"""
    pass

class ExtractionError(SDGPipelineError):
    """Content extraction errors"""
    pass

class VectorizationError(SDGPipelineError):
    """Vector database errors"""
    pass

def handle_errors(error_types: tuple = (Exception,), 
                 fallback_return: Any = None,
                 log_level: str = "error"):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                error_info = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200],
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                getattr(logger, log_level)(f"Error in {func.__name__}: {e}", extra=error_info)
                
                if isinstance(e, SDGPipelineError):
                    raise  # Re-raise custom errors
                
                return fallback_return
            except Exception as e:
                logger.critical(f"Unexpected error in {func.__name__}: {e}", 
                              extra={"traceback": traceback.format_exc()})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                error_info = {
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                getattr(logger, log_level)(f"Error in {func.__name__}: {e}", extra=error_info)
                return fallback_return
            except Exception as e:
                logger.critical(f"Unexpected error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Nach Zeile 100 hinzufügen:

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
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
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

weaviate_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
external_api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self):
        self.retry_strategies = {}
        
    def register_retry_strategy(self, error_type: Type[Exception], 
                              max_retries: int = 3, 
                              delay: float = 1.0,
                              backoff: float = 2.0):
        """Register retry strategy for specific error types"""
        self.retry_strategies[error_type] = {
            "max_retries": max_retries,
            "delay": delay,
            "backoff": backoff
        }
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for error_type, strategy in self.retry_strategies.items():
            max_retries = strategy["max_retries"]
            delay = strategy["delay"]
            backoff = strategy["backoff"]
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except error_type as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        break
        
        if last_exception:
            raise last_exception

# Global error recovery manager
error_manager = ErrorRecoveryManager()

# Register common retry strategies
error_manager.register_retry_strategy(DatabaseError, max_retries=3, delay=2.0)
error_manager.register_retry_strategy(ExtractionError, max_retries=2, delay=5.0)
error_manager.register_retry_strategy(VectorizationError, max_retries=2, delay=3.0)
