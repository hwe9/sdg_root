import os, errno, tempfile
import json
import yaml
from typing import Dict, Any, Optional, List
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
    # New dependency management fields
    dependency_health_check_interval: int = 60
    max_dependency_wait_time: int = 300
    required_dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    additional_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Central configuration management"""
    
    def __init__(self, config_dir: str = None):
        requested = Path(config_dir or os.getenv("CONFIG_DIR", "/app/config"))
        self._readonly = False
        try:
            requested.mkdir(parents=True, exist_ok=True)
            self.config_dir = requested
        except OSError as e:
            if e.errno in (errno.EROFS, errno.EACCES):
                fallback = Path(os.getenv("CONFIG_FALLBACK_DIR", "/tmp/sdg_config"))
                fallback.mkdir(parents=True, exist_ok=True)
                self.config_dir = fallback
                self._readonly = True
                logger.warning(f"Config dir {requested} not writable; using in-memory config with dir {self.config_dir}")
            else:
                raise
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

        try:
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
            
            # Service configurations with dependency mapping
            services = {
                "data_retrieval": {
                    "port": 8002, 
                    "timeout": 60,
                    "required_dependencies": ["database"],
                    "optional_dependencies": []
                },
                "data_processing": {
                    "port": 8001, 
                    "timeout": 120, 
                    "max_connections": 50,
                    "required_dependencies": ["database", "weaviate", "data_retrieval"],
                    "optional_dependencies": ["content_extraction"]
                },
                "vectorization": {
                    "port": 8003, 
                    "timeout": 30,
                    "required_dependencies": ["weaviate", "database"],
                    "optional_dependencies": []
                },
                "content_extraction": {
                    "port": 8004, 
                    "timeout": 45,
                    "required_dependencies": ["database"],
                    "optional_dependencies": []
                },
                "api": {
                    "port": 8000, 
                    "timeout": 30,
                    "required_dependencies": ["database", "auth"],
                    "optional_dependencies": ["vectorization", "content_extraction"]
                },
                "auth": {
                    "port": 8005, 
                    "timeout": 15,
                    "required_dependencies": ["database"],
                    "optional_dependencies": []
                }
            }
            
            if not self._readonly:
                with open(self.config_dir / "global.yaml", 'w') as f:
                   yaml.dump(global_config, f, default_flow_style=False)
                services_dir = self.config_dir / "services"
                services_dir.mkdir(exist_ok=True)
                for service_name, config in services.items():
                    with open(services_dir / f"{service_name}.yaml", 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
            else:
                logger.info("Readonly config mode: using in-memory defaults; no files written.")
            
            self.global_config = global_config
            for service_name, config in services.items():
                self.services_config[service_name] = ServiceConfig(
                    name=service_name,
                    **config
                )
            
            logger.info("Created default configuration files with dependency mapping")
            
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
    
    def get_dependency_chain(self, service_name: str) -> Dict[str, List[str]]:
        """Get complete dependency chain for a service"""
        config = self.get_service_config(service_name)
        return {
            "required": config.required_dependencies,
            "optional": config.optional_dependencies,
            "all": config.required_dependencies + config.optional_dependencies
        }
    
    def get_service_startup_order(self) -> List[str]:
        """Calculate optimal service startup order based on dependencies"""
        services = list(self.services_config.keys())
        ordered_services = []
        remaining_services = set(services)
        
        while remaining_services:
            # Find services with no unmet dependencies
            ready_services = []
            for service in remaining_services:
                config = self.get_service_config(service)
                unmet_deps = set(config.required_dependencies) - set(ordered_services)
                if not unmet_deps:
                    ready_services.append(service)
            
            if not ready_services:
                # Circular dependency or missing service - add remaining arbitrarily
                logger.warning("Potential circular dependency detected, adding remaining services")
                ready_services = list(remaining_services)
            
            # Add ready services to order
            for service in ready_services:
                ordered_services.append(service)
                remaining_services.remove(service)
        
        return ordered_services
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate all service dependencies"""
        issues = {}
        all_services = set(self.services_config.keys())
        
        for service_name, config in self.services_config.items():
            service_issues = []
            
            # Check required dependencies
            for dep in config.required_dependencies:
                if dep not in all_services and dep not in ["database", "weaviate"]:  # Allow external deps
                    service_issues.append(f"Missing required dependency: {dep}")
            
            # Check optional dependencies
            for dep in config.optional_dependencies:
                if dep not in all_services and dep not in ["database", "weaviate"]:
                    service_issues.append(f"Missing optional dependency: {dep}")
            
            if service_issues:
                issues[service_name] = service_issues
        
        return issues
    
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
