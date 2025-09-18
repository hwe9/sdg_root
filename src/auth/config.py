# /sdg_root/src/auth/config.py
import os
from enum import Enum
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    allowed_origins: list[str] = []
    jwt_algorithm: str = "RS256"
    password_min_length: int = 8
    rate_limit_per_minute: int = 60
    auth_rate_limit_per_minute: int = 5
    max_file_size_mb: int = 50
    allowed_file_types: list = [".pdf", ".txt", ".docx", ".csv"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_csv(cls, v):
        if isinstance(v, str) and v.strip() and not v.strip().startswith("["):
            return [s.strip() for s in v.split(",")]
        return v

settings = Settings()
