import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    host: str = "localhost"
    port: int = 8000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_connections: int = 100
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    additional_config: Dict[str, Any] = field(default_factory=dict)

class ConfigurationManager:
    """Central configuration management"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or os.getenv("CONFIG_DIR", "/app/config"))
        self.config_dir.mkdir(exist_ok=True)
        self.services_config = {}
        self.global_config = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files"""
        try:
            # Load global configuration
            global_config_file = self.config_dir / "global.yaml"
            if global_config_file.exists():
                with open(global_config_file, 'r') as f:
                    self.global_config = yaml.safe_load(f)
            
            # Load service-specific configurations
            services_dir = self.config_dir / "services"
            if services_dir.exists():
                for config_file in services_dir.glob("*.yaml"):
                    service_name = config_file.stem
                    with open(config_file, 'r') as f:
                        service_config = yaml.safe_load(f)
                        self.services_config[service_name] = ServiceConfig(
                            name=service_name,
                            **service_config
                        )
            
            logger.info(f"Loaded configuration for {len(self.services_config)} services")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            self._create_default_configurations()
    
    def _create_default_configurations(self):
        """Create default configuration files"""
        try:
            # Global configuration
            global_config = {
                "environment": os.getenv("ENVIRONMENT", "development"),
                "debug": os.getenv("DEBUG", "false").lower() == "true",
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "database": {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_recycle": 300
                },
                "security": {
                    "allowed_origins": ["http://localhost:3000"],
                    "jwt_expire_minutes": 30
                }
            }
            
            # Service configurations
            services = {
                "data_retrieval": {"port": 8002, "timeout": 60},
                "data_processing": {"port": 8001, "timeout": 120, "max_connections": 50},
                "vectorization": {"port": 8003, "timeout": 30},
                "content_extraction": {"port": 8004, "timeout": 45},
                "api": {"port": 8000, "timeout": 30},
                "auth": {"port": 8005, "timeout": 15}
            }
            
            # Save configurations
            with open(self.config_dir / "global.yaml", 'w') as f:
                yaml.dump(global_config, f, default_flow_style=False)
            
            services_dir = self.config_dir / "services"
            services_dir.mkdir(exist_ok=True)
            
            for service_name, config in services.items():
                with open(services_dir / f"{service_name}.yaml", 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            self.global_config = global_config
            for service_name, config in services.items():
                self.services_config[service_name] = ServiceConfig(
                    name=service_name,
                    **config
                )
            
            logger.info("Created default configuration files")
            
        except Exception as e:
            logger.error(f"Error creating default configurations: {e}")
    
    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for specific service"""
        if service_name in self.services_config:
            return self.services_config[service_name]
        
        # Return default configuration
        logger.warning(f"No configuration found for {service_name}, using defaults")
        return ServiceConfig(name=service_name)
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        return self.global_config.copy()
    
    def update_service_config(self, service_name: str, updates: Dict[str, Any]):
        """Update service configuration"""
        if service_name in self.services_config:
            config = self.services_config[service_name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.additional_config[key] = value
        else:
            self.services_config[service_name] = ServiceConfig(
                name=service_name,
                **updates
            )
    
    def get_database_url(self) -> str:
        """Get database URL with proper error handling"""
        try:
            # Try environment variable first
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                return db_url
            
            # Construct from individual components
            db_config = self.global_config.get("database", {})
            host = os.getenv("DB_HOST", db_config.get("host", "localhost"))
            port = os.getenv("DB_PORT", db_config.get("port", "5432"))
            name = os.getenv("DB_NAME", db_config.get("name", "sdg_pipeline"))
            user = os.getenv("DB_USER", db_config.get("user", "postgres"))
            password = os.getenv("DB_PASSWORD", db_config.get("password", "postgres"))
            
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
            
        except Exception as e:
            logger.error(f"Error constructing database URL: {e}")
            # Fallback URL
            return "postgresql://postgres:postgres@localhost:5432/sdg_pipeline"

# Global configuration manager
config_manager = ConfigurationManager()
