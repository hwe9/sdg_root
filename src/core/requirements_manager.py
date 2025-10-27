# src/core/requirements_manager.py
"""
Centralized requirements management for SDG project
This module ensures consistent dependency versions across all services
"""

import os
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DependencyVersion:
    """Represents a dependency with its version constraints"""
    name: str
    version: str
    extras: Optional[str] = None
    comment: Optional[str] = None

class RequirementsManager:
    """Centralized requirements management"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.requirements_dir = self.base_dir / "requirements"
        self.requirements_dir.mkdir(exist_ok=True)
        
        # Define core dependencies with consistent versions
        self.core_dependencies = {
            # Web Framework
            "fastapi": DependencyVersion("fastapi", "0.115.0", comment="Web framework"),
            "uvicorn": DependencyVersion("uvicorn[standard]", "0.30.0", comment="ASGI server"),
            
            # Database
            "sqlalchemy": DependencyVersion("sqlalchemy", "2.0.43", comment="ORM"),
            "psycopg2-binary": DependencyVersion("psycopg2-binary", "2.9.10", comment="PostgreSQL adapter"),
            "alembic": DependencyVersion("alembic", "1.13.1", comment="Database migrations"),
            
            # Authentication & Security
            "python-jose": DependencyVersion("python-jose[cryptography]", "3.3.0", comment="JWT handling"),
            "passlib": DependencyVersion("passlib[bcrypt]", "1.7.4", comment="Password hashing"),
            "cryptography": DependencyVersion("cryptography", "41.0.0", comment="Cryptographic utilities"),
            
            # HTTP & Networking
            "httpx": DependencyVersion("httpx", "0.28.1", comment="HTTP client"),
            "requests": DependencyVersion("requests", "2.31.0", comment="HTTP library"),
            
            # Data Processing
            "pydantic": DependencyVersion("pydantic", "2.8.0", comment="Data validation"),
            "python-multipart": DependencyVersion("python-multipart", "0.0.6", comment="Form data handling"),
            "python-dateutil": DependencyVersion("python-dateutil", "2.8.2", comment="Date utilities"),
            
            # AI/ML Dependencies
            "sentence-transformers": DependencyVersion("sentence-transformers", "3.0.1", comment="Text embeddings"),
            "transformers": DependencyVersion("transformers", "4.44.2", comment="Hugging Face transformers"),
            "torch": DependencyVersion("torch", "2.1.0", comment="PyTorch"),
            "numpy": DependencyVersion("numpy", "1.24.3", comment="Numerical computing"),
            "scikit-learn": DependencyVersion("scikit-learn", "1.3.0", comment="Machine learning"),
            
            # Vector Database
            "weaviate-client": DependencyVersion("weaviate-client", "4.16.10", comment="Weaviate client"),
            
            # Monitoring & Metrics
            "prometheus-client": DependencyVersion("prometheus-client", "0.20.0", comment="Prometheus metrics"),
            
            # Utilities
            "python-dotenv": DependencyVersion("python-dotenv", "1.0.0", comment="Environment variables"),
            "redis": DependencyVersion("redis", "5.0.1", comment="Redis client"),
            "celery": DependencyVersion("celery", "5.3.4", comment="Task queue"),
        }
        
        # Service-specific dependencies
        self.service_dependencies = {
            "api": [
                "fastapi", "uvicorn", "sqlalchemy", "psycopg2-binary", "pydantic",
                "python-multipart", "python-jose", "passlib", "httpx", "prometheus-client",
                "alembic", "python-dotenv", "redis", "celery"
            ],
            "auth": [
                "fastapi", "uvicorn", "sqlalchemy", "psycopg2-binary", "pydantic",
                "python-multipart", "python-jose", "passlib", "cryptography", "httpx",
                "prometheus-client", "python-dotenv", "redis", "bcrypt", "PyJWT",
                "pydantic-settings", "slowapi", "keyring", "APScheduler"
            ],
            "data_processing": [
                "faster-whisper", "sentence-transformers", "sqlalchemy", "psycopg2-binary",
                "pypdf2", "python-docx", "requests", "Pillow", "pytesseract",
                "deep_translator", "weaviate-client"
            ],
            "data_retrieval": [
                "fastapi", "uvicorn", "httpx", "requests", "pydantic", "python-multipart",
                "sqlalchemy", "psycopg2-binary", "prometheus-client", "python-dotenv"
            ],
            "content_extraction": [
                "fastapi", "uvicorn", "httpx", "requests", "pydantic", "python-multipart",
                "sqlalchemy", "psycopg2-binary", "prometheus-client", "python-dotenv",
                "openai", "google-generativeai", "beautifulsoup4", "feedparser"
            ],
            "vectorization": [
                "fastapi", "uvicorn", "sentence-transformers", "transformers", "torch",
                "numpy", "scikit-learn", "weaviate-client", "openai", "networkx",
                "asyncio-mqtt", "python-multipart", "pydantic", "python-jose",
                "httpx", "prometheus-client", "sqlalchemy", "psycopg2-binary"
            ]
        }
    
    def generate_requirements_file(self, service_name: str) -> str:
        """Generate requirements.txt content for a specific service"""
        if service_name not in self.service_dependencies:
            raise ValueError(f"Unknown service: {service_name}")
        
        dependencies = self.service_dependencies[service_name]
        lines = []
        
        # Add header comment
        lines.append(f"# Requirements for {service_name} service")
        lines.append(f"# Generated by RequirementsManager")
        lines.append("")
        
        # Add dependencies
        for dep_name in dependencies:
            if dep_name in self.core_dependencies:
                dep = self.core_dependencies[dep_name]
                line = f"{dep.name}=={dep.version}"
                if dep.comment:
                    line += f"  # {dep.comment}"
                lines.append(line)
            else:
                # Handle special cases
                if dep_name == "faster-whisper":
                    lines.append("faster-whisper  # Audio transcription")
                elif dep_name == "pypdf2":
                    lines.append("pypdf2  # PDF processing")
                elif dep_name == "python-docx":
                    lines.append("python-docx  # Word document processing")
                elif dep_name == "Pillow":
                    lines.append("Pillow  # Image processing")
                elif dep_name == "pytesseract":
                    lines.append("pytesseract  # OCR")
                elif dep_name == "deep_translator":
                    lines.append("deep_translator  # Translation")
                elif dep_name == "bcrypt":
                    lines.append("bcrypt>=4.0.0  # Password hashing")
                elif dep_name == "PyJWT":
                    lines.append("PyJWT[crypto]>=2.8.0  # JWT handling")
                elif dep_name == "pydantic-settings":
                    lines.append("pydantic-settings>=2.3.0  # Settings management")
                elif dep_name == "slowapi":
                    lines.append("slowapi==0.1.9  # Rate limiting")
                elif dep_name == "keyring":
                    lines.append("keyring>=24.0.0  # Key management")
                elif dep_name == "APScheduler":
                    lines.append("APScheduler>=3.10.4  # Task scheduling")
                elif dep_name == "openai":
                    lines.append("openai==1.40.0  # OpenAI API")
                elif dep_name == "google-generativeai":
                    lines.append("google-generativeai  # Google AI")
                elif dep_name == "beautifulsoup4":
                    lines.append("beautifulsoup4  # HTML parsing")
                elif dep_name == "feedparser":
                    lines.append("feedparser  # RSS parsing")
                elif dep_name == "networkx":
                    lines.append("networkx==3.2  # Graph processing")
                elif dep_name == "asyncio-mqtt":
                    lines.append("asyncio-mqtt==0.13.0  # MQTT client")
                else:
                    lines.append(f"{dep_name}  # Unknown dependency")
        
        return "\n".join(lines)
    
    def update_all_requirements(self) -> Dict[str, bool]:
        """Update requirements.txt files for all services"""
        results = {}
        
        for service_name in self.service_dependencies.keys():
            try:
                requirements_content = self.generate_requirements_file(service_name)
                requirements_file = self.base_dir / "src" / service_name / "requirements.txt"
                
                # Backup existing file
                if requirements_file.exists():
                    backup_file = requirements_file.with_suffix(".txt.backup")
                    requirements_file.rename(backup_file)
                    logger.info(f"Backed up {requirements_file} to {backup_file}")
                
                # Write new requirements file
                requirements_file.write_text(requirements_content)
                results[service_name] = True
                logger.info(f"Updated requirements for {service_name}")
                
            except Exception as e:
                logger.error(f"Failed to update requirements for {service_name}: {e}")
                results[service_name] = False
        
        return results
    
    def validate_requirements(self) -> Dict[str, List[str]]:
        """Validate that all requirements files are consistent"""
        issues = {}
        
        for service_name in self.service_dependencies.keys():
            service_issues = []
            requirements_file = self.base_dir / "src" / service_name / "requirements.txt"
            
            if not requirements_file.exists():
                service_issues.append("Requirements file does not exist")
                issues[service_name] = service_issues
                continue
            
            try:
                current_content = requirements_file.read_text()
                expected_content = self.generate_requirements_file(service_name)
                
                if current_content.strip() != expected_content.strip():
                    service_issues.append("Requirements file is not up to date")
                
                # Check for version inconsistencies
                lines = current_content.split('\n')
                for line in lines:
                    if '==' in line and not line.strip().startswith('#'):
                        dep_name = line.split('==')[0].strip()
                        if dep_name in self.core_dependencies:
                            expected_version = self.core_dependencies[dep_name].version
                            actual_version = line.split('==')[1].split()[0].strip()
                            if actual_version != expected_version:
                                service_issues.append(f"{dep_name}: expected {expected_version}, got {actual_version}")
                
            except Exception as e:
                service_issues.append(f"Error reading requirements file: {e}")
            
            if service_issues:
                issues[service_name] = service_issues
        
        return issues
    
    def generate_base_requirements(self) -> str:
        """Generate base requirements.txt for the project root"""
        lines = [
            "# SDG Project Base Requirements",
            "# Core dependencies used across all services",
            "",
        ]
        
        # Add core dependencies
        for dep_name, dep in self.core_dependencies.items():
            line = f"{dep.name}=={dep.version}"
            if dep.comment:
                line += f"  # {dep.comment}"
            lines.append(line)
        
        return "\n".join(lines)

# Global requirements manager instance
requirements_manager = RequirementsManager()

