# src/core/env_manager.py
"""
Centralized environment variable management for SDG project
This module ensures consistent environment variable handling across all services
"""

import os
import logging
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from typing import Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EnvVarConfig:
    """Configuration for an environment variable"""
    name: str
    default: Optional[str] = None
    required: bool = False
    description: str = ""
    validation_func: Optional[callable] = None
    service_specific: bool = False

class EnvironmentManager:
    """Centralized environment variable management"""
    
    def __init__(self):
        self.env_configs = self._define_env_configs()
        # Don't validate on initialization to allow template generation
        # self._validate_environment()
    
    def _define_env_configs(self) -> Dict[str, EnvVarConfig]:
        """Define all environment variable configurations"""
        return {
            # Database Configuration
            "DATABASE_URL": EnvVarConfig(
                name="DATABASE_URL",
                description="Primary database connection URL",
                validation_func=self._validate_database_url
            ),
            "DB_HOST": EnvVarConfig(
                name="DB_HOST",
                default="database_service",
                description="Database host"
            ),
            "DB_PORT": EnvVarConfig(
                name="DB_PORT",
                default="5432",
                description="Database port"
            ),
            "POSTGRES_USER": EnvVarConfig(
                name="POSTGRES_USER",
                default="postgres",
                description="PostgreSQL username"
            ),
            "POSTGRES_PASSWORD": EnvVarConfig(
                name="POSTGRES_PASSWORD",
                required=True,
                description="PostgreSQL password"
            ),
            "POSTGRES_DB": EnvVarConfig(
                name="POSTGRES_DB",
                default="sdg_pipeline",
                description="PostgreSQL database name"
            ),
            
            # Redis Configuration
            "REDIS_URL": EnvVarConfig(
                name="REDIS_URL",
                default="redis://redis:6379",
                description="Redis connection URL",
                validation_func=self._validate_redis_url
            ),
            "REDIS_PASSWORD": EnvVarConfig(
                name="REDIS_PASSWORD",
                description="Redis password"
            ),
            
            # Weaviate Configuration
            "WEAVIATE_URL": EnvVarConfig(
                name="WEAVIATE_URL",
                default="http://weaviate_service:8080",
                description="Weaviate vector database URL",
                validation_func=self._validate_weaviate_url
            ),
            "WEAVIATE_API_KEY": EnvVarConfig(
                name="WEAVIATE_API_KEY",
                description="Weaviate API key"
            ),
            "WEAVIATE_TRANSFORMER_URL": EnvVarConfig(
                name="WEAVIATE_TRANSFORMER_URL",
                default="http://weaviate_transformer_service:8080",
                description="Weaviate transformer service URL"
            ),
            
            # Security Configuration
            "SECRET_KEY": EnvVarConfig(
                name="SECRET_KEY",
                required=True,
                description="Secret key for JWT and encryption",
                validation_func=self._validate_secret_key
            ),
            "SECRET_KEY_ENCRYPTED": EnvVarConfig(
                name="SECRET_KEY_ENCRYPTED",
                description="Encrypted secret key"
            ),
            "ENCRYPTION_SALT": EnvVarConfig(
                name="ENCRYPTION_SALT",
                default="default_salt",
                description="Salt for encryption"
            ),
            
            # Service Configuration
            "ENVIRONMENT": EnvVarConfig(
                name="ENVIRONMENT",
                default="development",
                description="Environment (development, staging, production)",
                validation_func=self._validate_environment_type
            ),
            "LOG_LEVEL": EnvVarConfig(
                name="LOG_LEVEL",
                default="INFO",
                description="Logging level",
                validation_func=self._validate_log_level
            ),
            "DEBUG": EnvVarConfig(
                name="DEBUG",
                default="false",
                description="Debug mode",
                validation_func=self._validate_boolean
            ),
            
            # CORS Configuration
            "ALLOWED_ORIGINS": EnvVarConfig(
                name="ALLOWED_ORIGINS",
                default="*",
                description="Allowed CORS origins"
            ),
            
            # Service URLs
            "API_SERVICE_URL": EnvVarConfig(
                name="API_SERVICE_URL",
                default="http://api_service:8000",
                description="API service URL"
            ),
            "AUTH_SERVICE_URL": EnvVarConfig(
                name="AUTH_SERVICE_URL",
                default="http://auth_service:8005",
                description="Auth service URL"
            ),
            "DATA_PROCESSING_URL": EnvVarConfig(
                name="DATA_PROCESSING_URL",
                default="http://data_processing_service:8001",
                description="Data processing service URL"
            ),
            "DATA_RETRIEVAL_URL": EnvVarConfig(
                name="DATA_RETRIEVAL_URL",
                default="http://data_retrieval_service:8002",
                description="Data retrieval service URL"
            ),
            "VECTORIZATION_SERVICE_URL": EnvVarConfig(
                name="VECTORIZATION_SERVICE_URL",
                default="http://vectorization_service:8003",
                description="Vectorization service URL"
            ),
            "CONTENT_EXTRACTION_URL": EnvVarConfig(
                name="CONTENT_EXTRACTION_URL",
                default="http://content_extraction_service:8004",
                description="Content extraction service URL"
            ),
            
            # Dependency Management
            "DEPENDENCY_MANAGER_ENABLED": EnvVarConfig(
                name="DEPENDENCY_MANAGER_ENABLED",
                default="true",
                description="Enable dependency manager",
                validation_func=self._validate_boolean
            ),
            "HEALTH_CHECK_INTERVAL": EnvVarConfig(
                name="HEALTH_CHECK_INTERVAL",
                default="60",
                description="Health check interval in seconds",
                validation_func=self._validate_positive_int
            ),
            "MAX_STARTUP_TIME": EnvVarConfig(
                name="MAX_STARTUP_TIME",
                default="300",
                description="Maximum startup time in seconds",
                validation_func=self._validate_positive_int
            ),
            
            # Configuration Directories
            "CONFIG_DIR": EnvVarConfig(
                name="CONFIG_DIR",
                default="/data/config",
                description="Configuration directory path"
            ),
            "SECRET_STORE_DIR": EnvVarConfig(
                name="SECRET_STORE_DIR",
                default="/data/secrets",
                description="Secret store directory path"
            ),
            
            # Data Directories
            "DATA_ROOT": EnvVarConfig(
                name="DATA_ROOT",
                default="/data",
                description="Data root directory"
            ),
            "RAW_DATA_DIR": EnvVarConfig(
                name="RAW_DATA_DIR",
                default="/data/raw_data",
                description="Raw data directory"
            ),
            "PROCESSED_DATA_DIR": EnvVarConfig(
                name="PROCESSED_DATA_DIR",
                default="/data/processed_data",
                description="Processed data directory"
            ),
            "IMAGES_DIR": EnvVarConfig(
                name="IMAGES_DIR",
                default="/data/images",
                description="Images directory"
            ),
            
            # PgAdmin Configuration
            "PGADMIN_DEFAULT_EMAIL": EnvVarConfig(
                name="PGADMIN_DEFAULT_EMAIL",
                default="admin@sdg.local",
                description="PgAdmin default email"
            ),
            "PGADMIN_DEFAULT_PASSWORD": EnvVarConfig(
                name="PGADMIN_DEFAULT_PASSWORD",
                required=True,
                description="PgAdmin default password"
            ),
        }
    
    def get_env_var(self, name: str, default: Optional[str] = None) -> str:
        """Get environment variable with validation"""
        if name not in self.env_configs:
            logger.warning(f"Unknown environment variable: {name}")
            return os.getenv(name, default or "")
        
        config = self.env_configs[name]
        value = os.getenv(name, config.default)
        
        if config.required and not value:
            raise ValueError(f"Required environment variable {name} is not set")
        
        if value and config.validation_func:
            try:
                config.validation_func(value)
            except Exception as e:
                raise ValueError(f"Invalid value for {name}: {e}")
        
        return value or ""
    
    def get_all_env_vars(self) -> Dict[str, str]:
        """Get all environment variables with their values"""
        result = {}
        for name, config in self.env_configs.items():
            try:
                value = self.get_env_var(name)
                result[name] = value
            except ValueError as e:
                logger.error(f"Environment variable error: {e}")
                result[name] = f"ERROR: {e}"
        return result
    
    def _validate_environment(self):
        """Validate the current environment configuration"""
        errors = []
        
        for name, config in self.env_configs.items():
            try:
                value = self.get_env_var(name)
                if config.validation_func and value:
                    config.validation_func(value)
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        if errors:
            logger.error("Environment validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Environment validation failed: {len(errors)} errors")
    
    # Validation functions
    def _validate_database_url(self, value: str) -> bool:
        """Validate database URL format"""
        if not value.startswith(('postgresql://', 'postgres://')):
            raise ValueError("Database URL must start with postgresql:// or postgres://")
        return True
    
    def _validate_redis_url(self, value: str) -> bool:
        """Validate Redis URL format"""
        if not value.startswith('redis://'):
            raise ValueError("Redis URL must start with redis://")
        return True
    
    def _validate_weaviate_url(self, value: str) -> bool:
        """Validate Weaviate URL format"""
        if not value.startswith(('http://', 'https://')):
            raise ValueError("Weaviate URL must start with http:// or https://")
        return True
    
    def _validate_secret_key(self, value: str) -> bool:
        """Validate secret key"""
        if len(value) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return True
    
    def _validate_environment_type(self, value: str) -> bool:
        """Validate environment type"""
        if value not in ['development', 'staging', 'production']:
            raise ValueError("Environment must be development, staging, or production")
        return True
    
    def _validate_log_level(self, value: str) -> bool:
        """Validate log level"""
        if value.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return True
    
    def _validate_boolean(self, value: str) -> bool:
        """Validate boolean value"""
        if value.lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
            raise ValueError("Boolean value must be true/false, 1/0, or yes/no")
        return True
    
    def _validate_positive_int(self, value: str) -> bool:
        """Validate positive integer"""
        try:
            int_val = int(value)
            if int_val <= 0:
                raise ValueError("Value must be a positive integer")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Value must be a valid integer")
            raise
        return True
    
    def generate_env_template(self) -> str:
        """Generate environment variable template"""
        lines = [
            "# SDG Project Environment Variables",
            "# Copy this file to .env and fill in the values",
            "",
        ]
        
        # Group variables by category
        categories = {
            "Database": ["DATABASE_URL", "DB_HOST", "DB_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"],
            "Redis": ["REDIS_URL", "REDIS_PASSWORD"],
            "Weaviate": ["WEAVIATE_URL", "WEAVIATE_API_KEY", "WEAVIATE_TRANSFORMER_URL"],
            "Security": ["SECRET_KEY", "SECRET_KEY_ENCRYPTED", "ENCRYPTION_SALT"],
            "Service Configuration": ["ENVIRONMENT", "LOG_LEVEL", "DEBUG", "ALLOWED_ORIGINS"],
            "Service URLs": ["API_SERVICE_URL", "AUTH_SERVICE_URL", "DATA_PROCESSING_URL", "DATA_RETRIEVAL_URL", "VECTORIZATION_SERVICE_URL", "CONTENT_EXTRACTION_URL"],
            "Dependency Management": ["DEPENDENCY_MANAGER_ENABLED", "HEALTH_CHECK_INTERVAL", "MAX_STARTUP_TIME"],
            "Directories": ["CONFIG_DIR", "SECRET_STORE_DIR", "DATA_ROOT", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "IMAGES_DIR"],
            "PgAdmin": ["PGADMIN_DEFAULT_EMAIL", "PGADMIN_DEFAULT_PASSWORD"],
        }
        
        for category, var_names in categories.items():
            lines.append(f"# {category}")
            for var_name in var_names:
                if var_name in self.env_configs:
                    config = self.env_configs[var_name]
                    if config.required:
                        lines.append(f"{var_name}=  # REQUIRED: {config.description}")
                    else:
                        default = config.default or ""
                        lines.append(f"{var_name}={default}  # {config.description}")
            lines.append("")
        
        return "\n".join(lines)

# Global environment manager instance
env_manager = EnvironmentManager()
