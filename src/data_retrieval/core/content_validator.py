"""
Security & Quality Content Validation
Validates downloaded content for security and quality
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import magic
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""
    quality_score: float = 0.0
    content_type: str = ""
    file_size: int = 0
    checksum: str = ""

class ContentValidator:
    """
    Validates downloaded content for security and quality
    """
    
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_mime_types = {
            'application/pdf',
            'text/html',
            'text/plain', 
            'application/xml',
            'text/xml',
            'application/json'
        }
        self.malicious_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'onload=',
            b'onclick='
        ]
    
    async def initialize(self):
        """Initialize content validator"""
        try:
            # Test libmagic
            magic.from_buffer(b"test", mime=True)
            logger.info("✅ Content validator initialized")
        except Exception as e:
            logger.warning(f"⚠️ libmagic not available, using basic validation: {e}")
    
    async def validate_content(self, download_result: Dict[str, Any]) -> ValidationResult:
        """Comprehensive content validation"""
        try:
            file_path = download_result.get("file_path")
            if not file_path or not os.path.exists(file_path):
                return ValidationResult(
                    is_valid=False,
                    reason="File does not exist"
                )
            
            # File size validation
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return ValidationResult(
                    is_valid=False,
                    reason=f"File too large: {file_size} bytes",
                    file_size=file_size
                )
            
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    reason="Empty file",
                    file_size=file_size
                )
            
            # MIME type validation
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except:
                # Fallback to extension-based detection
                mime_type = download_result.get("content_type", "application/octet-stream")
            
            if mime_type not in self.allowed_mime_types:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Disallowed MIME type: {mime_type}",
                    content_type=mime_type,
                    file_size=file_size
                )
            
            # Content security validation
            with open(file_path, 'rb') as f:
                content_sample = f.read(8192)  # Read first 8KB
            
            # Check for malicious patterns
            for pattern in self.malicious_patterns:
                if pattern in content_sample.lower():
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Potentially malicious content detected",
                        content_type=mime_type,
                        file_size=file_size
                    )
            
            # Generate checksum
            checksum = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    checksum.update(chunk)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                file_size, mime_type, download_result
            )
            
            return ValidationResult(
                is_valid=True,
                reason="Content validation passed",
                quality_score=quality_score,
                content_type=mime_type,
                file_size=file_size,
                checksum=checksum.hexdigest()
            )
            
        except Exception as e:
            logger.error(f"Error validating content: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}"
            )
    
    def _calculate_quality_score(self, file_size: int, mime_type: str, 
                               download_result: Dict[str, Any]) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # File size score (0-0.3)
        if file_size > 10000:  # 10KB+
            score += 0.3
        elif file_size > 1000:  # 1KB+
            score += 0.15
        
        # MIME type score (0-0.2)
        if mime_type == 'application/pdf':
            score += 0.2
        elif mime_type in ['text/html', 'application/xml']:
            score += 0.15
        
        # Download method score (0-0.2)
        method = download_result.get("download_method", "")
        if method == "http":
            score += 0.2
        elif method == "youtube":
            score += 0.15
        
        # Has metadata score (0-0.3)
        if download_result.get("metadata"):
            score += 0.3
        elif download_result.get("title"):
            score += 0.15
        
        return min(score, 1.0)
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for content validator"""
        try:
            # Test libmagic
            test_result = magic.from_buffer(b"test content", mime=True)
            return {
                "status": "healthy",
                "libmagic_available": True,
                "max_file_size": self.max_file_size,
                "allowed_mime_types": list(self.allowed_mime_types)
            }
        except:
            return {
                "status": "healthy",
                "libmagic_available": False,
                "max_file_size": self.max_file_size,
                "allowed_mime_types": list(self.allowed_mime_types)
            }
