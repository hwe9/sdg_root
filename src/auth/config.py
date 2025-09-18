# /sdg_root/src/auth/config.py
import os
import json
from enum import Enum
from typing import List, Any
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
        extra="ignore",
        env_aliases={"allowed_origins": ["ALLOWED_ORIGINS", "ALLOWEDORIGINS"]},
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any):
        # None -> []
        if v is None:
            return []
        # Already a sequence
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        # String input: allow CSV or JSON list; tolerate empty string
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    data = json.loads(s)
                    return [str(x).strip() for x in data if str(x).strip()]
                except Exception:
                    # Fall through to CSV parsing
                    pass
            return [x.strip() for x in s.split(",") if x.strip()]
        # Fallback: coerce to list[str] or empty
        try:
            return [str(v).strip()] if str(v).strip() else []
        except Exception:
            return []

settings = Settings()
