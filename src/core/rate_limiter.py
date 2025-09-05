"""
Advanced Rate Limiting with Redis backend
Implements sliding window and token bucket algorithms
"""
import redis
import time
import json
import hashlib
from typing import Optional, Dict, Any
from functools import wraps
from fastapi import HTTPException, Request
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Redis-based rate limiter with multiple algorithms"""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
    def sliding_window_limit(self, 
                           key: str, 
                           limit: int, 
                           window_seconds: int,
                           identifier: str = None) -> bool:
        """Sliding window rate limiting"""
        
        if identifier:
            key = f"rate_limit:{key}:{identifier}"
        else:
            key = f"rate_limit:{key}"
        
        now = time.time()
        pipeline = self.redis_client.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window_seconds)
        
        # Count current requests in window
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window_seconds)
        
        results = pipeline.execute()
        request_count = results[1]
        
        return request_count < limit
    
    def token_bucket_limit(self,
                         key: str,
                         capacity: int,
                         refill_rate: float,
                         requested_tokens: int = 1) -> bool:
        """Token bucket rate limiting"""
        
        bucket_key = f"token_bucket:{key}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = self.redis_client.hgetall(bucket_key)
        
        if bucket_data:
            tokens = float(bucket_data.get("tokens", capacity))
            last_refill = float(bucket_data.get("last_refill", now))
        else:
            tokens = capacity
            last_refill = now
        
        # Calculate tokens to add
        time_elapsed = now - last_refill
        tokens_to_add = time_elapsed * refill_rate
        tokens = min(capacity, tokens + tokens_to_add)
        
        # Check if request can be served
        if tokens >= requested_tokens:
            tokens -= requested_tokens
            
            # Update bucket state
            self.redis_client.hset(bucket_key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            self.redis_client.expire(bucket_key, 3600)  # 1 hour expiry
            
            return True
        else:
            # Update last_refill even if request denied
            self.redis_client.hset(bucket_key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            return False

# Rate limiting decorators
def rate_limit(limit: int = 60, window: int = 60, algorithm: str = "sliding_window"):
    """Rate limiting decorator for FastAPI endpoints"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            
            rate_limiter = get_rate_limiter()  # DI durch dependency manager
            
            # Generate identifier from IP + user agent
            identifier = f"{request.client.host}:{hash(request.headers.get('user-agent', ''))}"
            
            if algorithm == "sliding_window":
                allowed = rate_limiter.sliding_window_limit(
                    key=func.__name__,
                    limit=limit,
                    window_seconds=window,
                    identifier=identifier
                )
            else:  # token_bucket
                allowed = rate_limiter.token_bucket_limit(
                    key=f"{func.__name__}:{identifier}",
                    capacity=limit,
                    refill_rate=limit / window
                )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {identifier} on {func.__name__}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {limit} requests per {window} seconds.",
                    headers={"Retry-After": str(window)}
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")
    return _rate_limiter
