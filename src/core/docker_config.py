# src/core/docker_config.py
"""
Centralized Docker configuration management for SDG project
This module ensures consistent Docker configurations across all services
"""

import os
import logging
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DockerServiceConfig:
    """Configuration for a Docker service"""
    name: str
    build_context: str
    dockerfile: str
    ports: List[str]
    environment: Dict[str, str]
    volumes: List[str]
    depends_on: List[str]
    networks: List[str]
    healthcheck: Optional[Dict[str, Any]] = None
    deploy: Optional[Dict[str, Any]] = None
    restart: str = "unless-stopped"

class DockerConfigManager:
    """Centralized Docker configuration management"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.services = self._define_services()
    
    def _define_services(self) -> Dict[str, DockerServiceConfig]:
        """Define all Docker service configurations"""
        return {
            "nginx_proxy": DockerServiceConfig(
                name="nginx_proxy",
                build_context="nginx",
                dockerfile="nginx.conf",
                ports=["80:80", "443:443"],
                environment={
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${SRC_ROOT}/nginx/nginx.conf:/etc/nginx/nginx.conf:ro",
                    "${SRC_ROOT}/nginx/ssl:/etc/ssl/certs:ro",
                    "${SRC_ROOT}/nginx/logs:/var/log/nginx"
                ],
                depends_on=["auth_service", "api_service"],
                networks=["sdg_external", "sdg_internal"],
                healthcheck={
                    "test": ["CMD", "curl", "-fsS", "http://localhost:80/health"],
                    "interval": "30s",
                    "timeout": "15s",
                    "retries": "5",
                    "start_period": "60s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "0.5",
                            "memory": "256M"
                        }
                    }
                }
            ),
            
            "auth_service": DockerServiceConfig(
                name="auth_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/auth/Dockerfile",
                ports=["8005"],
                environment={
                    "PYTHONPATH": "/app",
                    "CONFIG_DIR": "/data/config",
                    "REDIS_URL": "${REDIS_URL}",
                    "SECRET_KEY": "${SECRET_KEY}",
                    "SECRET_KEY_ENCRYPTED": "${SECRET_KEY_ENCRYPTED:-}",
                    "DATABASE_URL": "${DATABASE_URL}",
                    "DATABASE_URL_ENCRYPTED": "${DATABASE_URL_ENCRYPTED:-}",
                    "ALLOWED_ORIGINS": "${ALLOWED_ORIGINS}",
                    "ENVIRONMENT": "${ENVIRONMENT}",
                    "SERVICE_NAME": "auth",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300",
                    "CONTENT_EXTRACTION_URL": "http://content_extraction_service:8004"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro",
                    "${DATA_ROOT}/secrets:/data/secrets",
                    "${DATA_ROOT}/config:/data/config"
                ],
                depends_on=["database_service", "db_bootstrap", "redis", "content_extraction_service"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://localhost:8005/health"],
                    "interval": "30s",
                    "timeout": "15s",
                    "retries": "5",
                    "start_period": "60s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "0.5",
                            "memory": "512M"
                        }
                    }
                }
            ),
            
            "api_service": DockerServiceConfig(
                name="api_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/api/Dockerfile",
                ports=["8000:8000"],
                environment={
                    "PYTHONPATH": "/app",
                    "DATABASE_URL": "${DATABASE_URL}",
                    "REDIS_URL": "${REDIS_URL}",
                    "WEAVIATE_URL": "${WEAVIATE_URL}",
                    "ENVIRONMENT": "${ENVIRONMENT:-development}",
                    "LOG_LEVEL": "INFO",
                    "WORKERS": "4",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro"
                ],
                depends_on=["database_service", "db_bootstrap", "auth_service", "weaviate_service", "weaviate_transformer_service", "redis"],
                networks=["sdg_internal", "sdg_external"],
                healthcheck={
                    "test": ["CMD", "curl", "-fsS", "http://localhost:8000/live"],
                    "interval": "15s",
                    "timeout": "5s",
                    "retries": "10",
                    "start_period": "30s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "2.0",
                            "memory": "2G"
                        },
                        "reservations": {
                            "cpus": "1.0",
                            "memory": "1G"
                        }
                    }
                }
            ),
            
            "database_service": DockerServiceConfig(
                name="database_service",
                build_context="postgres:16-alpine",
                dockerfile="",
                ports=[],
                environment={
                    "POSTGRES_USER": "${POSTGRES_USER:-postgres}",
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
                    "POSTGRES_DB": "${POSTGRES_DB:-sdg_pipeline}",
                    "POSTGRES_INITDB_ARGS": "--encoding=UTF-8 --locale=en_US.UTF-8",
                    "POSTGRES_SHARED_PRELOAD_LIBRARIES": "pg_stat_statements",
                    "POSTGRES_MAX_CONNECTIONS": "200",
                    "POSTGRES_SHARED_BUFFERS": "256MB",
                    "POSTGRES_EFFECTIVE_CACHE_SIZE": "1GB",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${DATABASE_SERVICE_VOLUMES}",
                    "${SRC_ROOT}/postgres/init:/docker-entrypoint-initdb.d:ro",
                    "${SRC_ROOT}/postgres/backup:/backup"
                ],
                depends_on=[],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": "5",
                    "start_period": "30s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "2.0",
                            "memory": "4G"
                        },
                        "reservations": {
                            "cpus": "1.0",
                            "memory": "2G"
                        }
                    }
                }
            ),
            
            "redis": DockerServiceConfig(
                name="redis",
                build_context="redis:7-alpine",
                dockerfile="",
                ports=["127.0.0.1:6379:6379"],
                environment={
                    "REDIS_ARGS": "--requirepass ${REDIS_PASSWORD}"
                },
                volumes=[
                    "redis_data:/data",
                    "${SRC_ROOT}/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro"
                ],
                depends_on=[],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD-SHELL", "redis-cli -a ${REDIS_PASSWORD} -h 127.0.0.1 -p 6379 ping | grep PONG"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "0.5",
                            "memory": "512M"
                        }
                    }
                }
            ),
            
            "weaviate_service": DockerServiceConfig(
                name="weaviate_service",
                build_context="semitechnologies/weaviate:1.24.1",
                dockerfile="",
                ports=["8080:8080", "50051:50051"],
                environment={
                    "QUERY_DEFAULTS_LIMIT": "25",
                    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                    "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
                    "DEFAULT_VECTORIZER_MODULE": "none",
                    "ENABLE_MODULES": "text2vec-transformers",
                    "TRANSFORMERS_INFERENCE_API": "http://weaviate_transformer_service:8080",
                    "CLUSTER_HOSTNAME": "node1",
                    "DISABLE_TELEMETRY": "true"
                },
                volumes=[
                    "${WEAVIATE_SERVICE_VOLUMES}"
                ],
                depends_on=["weaviate_transformer_service"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": [
                        "CMD-SHELL",
                        "(command -v curl >/dev/null 2>&1 && (curl -fsS http://localhost:8080/v1/.well-known/ready >/dev/null || curl -fsS http://localhost:8080/.well-known/ready >/dev/null)) || (command -v wget >/dev/null 2>&1 && (wget -q -O - http://localhost:8080/v1/.well-known/ready >/dev/null || wget -q -O - http://localhost:8080/.well-known/ready >/dev/null)) || exit 1"
                    ],
                    "interval": "20s",
                    "timeout": "10s",
                    "retries": "20",
                    "start_period": "120s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "2.0",
                            "memory": "2G"
                        }
                    }
                }
            ),
            
            "weaviate_transformer_service": DockerServiceConfig(
                name="weaviate_transformer_service",
                build_context="${WEAVIATE_IMAGE}",
                dockerfile="",
                ports=["127.0.0.1:8081:8080"],
                environment={
                    "ENABLE_GPU": "0",
                    "TORCH_NUM_THREADS": "4",
                    "OMP_NUM_THREADS": "4"
                },
                volumes=[],
                depends_on=[],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD-SHELL", "python - <<'PY'\nimport urllib.request,sys\ntry:\n  urllib.request.urlopen('http://localhost:8080/.well-known/ready', timeout=5).read()\n  sys.exit(0)\nexcept Exception:\n  sys.exit(1)\nPY"],
                    "interval": "45s",
                    "timeout": "30s",
                    "retries": "25",
                    "start_period": "600s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "4.0",
                            "memory": "8G"
                        },
                        "reservations": {
                            "memory": "4G"
                        }
                    }
                }
            ),
            
            "data_processing_service": DockerServiceConfig(
                name="data_processing_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/data_processing/Dockerfile",
                ports=["8001:8001"],
                environment={
                    "PYTHONPATH": "/app",
                    "DATABASE_URL": "${DATABASE_URL:-}",
                    "RAW_DATA_DIR": "/data/raw_data",
                    "PROCESSED_DATA_DIR": "/data/processed_data",
                    "IMAGES_DIR": "/data/images",
                    "WEAVIATE_URL": "${WEAVIATE_URL}",
                    "WEAVIATE_API_KEY": "${WEAVIATE_API_KEY}",
                    "REDIS_URL": "redis://redis:6379",
                    "ENVIRONMENT": "${ENVIRONMENT:-development}",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro",
                    "${DATA_ROOT}/raw_data:/data/raw_data",
                    "${DATA_ROOT}/processed_data:/data/processed_data",
                    "${DATA_ROOT}/images:/data/images"
                ],
                depends_on=["database_service", "db_bootstrap", "weaviate_service", "data_retrieval_service"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://localhost:8001/health"],
                    "interval": "45s",
                    "timeout": "20s",
                    "retries": "5",
                    "start_period": "120s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "4.0",
                            "memory": "4G"
                        },
                        "reservations": {
                            "cpus": "2",
                            "memory": "3G"
                        }
                    }
                }
            ),
            
            "vectorization_service": DockerServiceConfig(
                name="vectorization_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/vectorization/Dockerfile",
                ports=["8003:8003"],
                environment={
                    "PYTHONPATH": "/app",
                    "WEAVIATE_URL": "${WEAVIATE_URL}",
                    "WEAVIATE_API_KEY": "${WEAVIATE_API_KEY}",
                    "DATABASE_URL": "${DATABASE_URL:-}",
                    "REDIS_URL": "${REDIS_URL}",
                    "ENVIRONMENT": "${ENVIRONMENT:-development}",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro"
                ],
                depends_on=["database_service", "weaviate_service"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://localhost:8003/health"],
                    "interval": "30s",
                    "timeout": "15s",
                    "retries": "5",
                    "start_period": "120s"
                }
            ),
            
            "data_retrieval_service": DockerServiceConfig(
                name="data_retrieval_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/data_retrieval/Dockerfile",
                ports=["8002:8002"],
                environment={
                    "PYTHONPATH": "/app",
                    "DATA_DIR": "/data/raw_data",
                    "PROCESSED_FILE": "/data/processed_data/processed_data.json",
                    "SOURCES_FILE": "/app/src/data_retrieval/new_sources.txt",
                    "SERVICE_HOST": "data_retrieval_service",
                    "SERVICE_PORT": "8002",
                    "DATABASE_URL": "${DATABASE_URL:-}",
                    "ALLOWED_ORIGINS": "${ALLOWED_ORIGINS}",
                    "SECRET_KEY": "${SECRET_KEY}",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro",
                    "${DATA_ROOT}/raw_data:/data/raw_data",
                    "${DATA_ROOT}/processed_data:/data/processed_data"
                ],
                depends_on=["database_service", "db_bootstrap"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD", "curl", "-f", "http://localhost:8002/health"],
                    "interval": "30s",
                    "timeout": "15s",
                    "retries": "5",
                    "start_period": "60s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "1.0"
                        }
                    }
                }
            ),
            
            "content_extraction_service": DockerServiceConfig(
                name="content_extraction_service",
                build_context="${SRC_ROOT}",
                dockerfile="src/content_extraction/Dockerfile",
                ports=["8004"],
                environment={
                    "PYTHONPATH": "/app",
                    "CONFIG_DIR": "/data/config",
                    "SECRET_STORE_DIR": "/data/secrets",
                    "DATABASE_URL": "${DATABASE_URL:-postgresql://postgres:postgres@database_service:5432/sdg_pipeline}",
                    "WEAVIATE_URL": "${WEAVIATE_URL}",
                    "WEAVIATE_TRANSFORMER_URL": "http://weaviate_transformer_service:8080",
                    "REDIS_URL": "redis://redis:6379",
                    "ENVIRONMENT": "${ENVIRONMENT:-production}",
                    "SECRET_KEY": "${SECRET_KEY}",
                    "DEPENDENCY_MANAGER_ENABLED": "true",
                    "HEALTH_CHECK_INTERVAL": "60",
                    "MAX_STARTUP_TIME": "300",
                    "LOG_LEVEL": "${LOG_LEVEL:-INFO}"
                },
                volumes=[
                    "${SRC_ROOT}:/app:ro",
                    "${DATA_ROOT}/config:/data/config",
                    "${DATA_ROOT}/secrets:/data/secrets"
                ],
                depends_on=["database_service", "db_bootstrap"],
                networks=["sdg_internal"],
                healthcheck={
                    "test": ["CMD-SHELL", "curl -fsS http://localhost:8004/health || exit 1"],
                    "interval": "30s",
                    "timeout": "15s",
                    "retries": "5",
                    "start_period": "60s"
                },
                deploy={
                    "resources": {
                        "limits": {
                            "cpus": "1.0",
                            "memory": "2G"
                        }
                    }
                }
            )
        }
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content"""
        lines = [
            "networks:",
            "  sdg_internal:",
            "    driver: bridge",
            "    internal: true",
            "    ipam:",
            "      driver: default",
            "      config:",
            "        - subnet: 172.20.0.0/16",
            "  sdg_external:",
            "    driver: bridge",
            "    ipam:",
            "      driver: default",
            "      config:",
            "        - subnet: 172.21.0.0/16",
            "  monitoring_network:",
            "    driver: bridge",
            "    internal: true",
            "",
            "services:"
        ]
        
        for service_name, config in self.services.items():
            lines.append(f"  {service_name}:")
            
            if config.build_context.startswith("${") or "/" in config.build_context:
                lines.append(f"    build:")
                lines.append(f"      context: {config.build_context}")
                if config.dockerfile:
                    lines.append(f"      dockerfile: {config.dockerfile}")
            else:
                lines.append(f"    image: {config.build_context}")
            
            if config.ports:
                lines.append(f"    ports:")
                for port in config.ports:
                    lines.append(f"      - \"{port}\"")
            
            if config.environment:
                lines.append(f"    environment:")
                for key, value in config.environment.items():
                    lines.append(f"      - {key}={value}")
            
            if config.volumes:
                lines.append(f"    volumes:")
                for volume in config.volumes:
                    lines.append(f"      - {volume}")
            
            if config.depends_on:
                lines.append(f"    depends_on:")
                for dep in config.depends_on:
                    lines.append(f"      - {dep}")
            
            if config.networks:
                lines.append(f"    networks:")
                for network in config.networks:
                    lines.append(f"      - {network}")
            
            if config.healthcheck:
                lines.append(f"    healthcheck:")
                for key, value in config.healthcheck.items():
                    if isinstance(value, list):
                        lines.append(f"      {key}:")
                        for item in value:
                            lines.append(f"        - \"{item}\"")
                    else:
                        lines.append(f"      {key}: {value}")
            
            if config.deploy:
                lines.append(f"    deploy:")
                for key, value in config.deploy.items():
                    lines.append(f"      {key}:")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            lines.append(f"        {sub_key}:")
                            if isinstance(sub_value, dict):
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    lines.append(f"          {sub_sub_key}: \"{sub_sub_value}\"")
                            else:
                                lines.append(f"          {sub_value}")
                    else:
                        lines.append(f"        {value}")
            
            lines.append(f"    restart: {config.restart}")
            lines.append("")
        
        # Add volumes section
        lines.extend([
            "volumes:",
            "  redis_data:",
            "    driver: local",
            "",
            "secrets:",
            "  postgres_password:",
            "    file: ${SRC_ROOT}/secrets/postgres_password.txt",
            "  app_db_password:",
            "    file: ${SRC_ROOT}/secrets/app_db_password.txt",
            "  weaviate_api_key:",
            "    file: ${SRC_ROOT}/secrets/weaviate_api_key.txt"
        ])
        
        return "\n".join(lines)
    
    def validate_docker_config(self) -> Dict[str, List[str]]:
        """Validate Docker configuration consistency"""
        issues = {}
        
        # Check for missing dependencies
        for service_name, config in self.services.items():
            service_issues = []
            
            for dep in config.depends_on:
                if dep not in self.services:
                    service_issues.append(f"Missing dependency: {dep}")
            
            if service_issues:
                issues[service_name] = service_issues
        
        return issues

# Global Docker config manager instance
docker_config_manager = DockerConfigManager()

