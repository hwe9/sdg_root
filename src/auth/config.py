# /sdg_root/src/auth/config.py
import os
from enum import Enum
from pydantic import BaseSettings

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Security settings
    allowed_origins: list = []
    jwt_algorithm: str = "RS256"
    password_min_length: int = 8
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    auth_rate_limit_per_minute: int = 5
    
    # File upload limits
    max_file_size_mb: int = 50
    allowed_file_types: list = [".pdf", ".txt", ".docx", ".csv"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
