"""
Protocol-specific download strategies
Handles different content types and protocols securely
"""
import os
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
import aiofiles
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadStrategy:
    """Base class for download strategies"""
    
    def __init__(self):
        self.session = None
        self.timeout = 30
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.user_agent = 'SDG-Pipeline-Bot/2.0 (+https://sdg-pipeline.org/bot)'
    
    async def initialize(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.user_agent}
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    def generate_safe_filename(self, url: str, content_type: str = "") -> str:
        """Generate safe filename from URL and content type"""
        parsed = urlparse(url)
        path_part = parsed.path.split('/')[-1] if parsed.path else 'download'
        
        # Sanitize filename
        safe_chars = '-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        safe_filename = ''.join(c for c in path_part if c in safe_chars)
        
        if not safe_filename or len(safe_filename) < 3:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            safe_filename = f"download_{url_hash}"
        
        # Add appropriate extension based on content type
        if not '.' in safe_filename:
            if 'pdf' in content_type:
                safe_filename += '.pdf'
            elif 'html' in content_type:
                safe_filename += '.html'
            elif 'xml' in content_type:
                safe_filename += '.xml'
            else:
                safe_filename += '.txt'
        
        return safe_filename

class HTTPDownloadStrategy(DownloadStrategy):
    """HTTP/HTTPS download with security validation"""
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Download HTTP/HTTPS content securely"""
        try:
            # Make request with security headers
            async with self.session.get(
                url,
                allow_redirects=True,
                headers={
                    'Accept': 'application/pdf,text/html,application/xml,text/plain',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'close'
                }
            ) as response:
                response.raise_for_status()
                
                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                allowed_types = ['application/pdf', 'text/html', 'text/plain', 'application/xml']
                
                if not any(allowed_type in content_type for allowed_type in allowed_types):
                    logger.warning(f"Disallowed content type: {content_type}")
                    return None
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    logger.warning(f"File too large: {content_length} bytes")
                    return None
                
                # Generate filename
                filename = self.generate_safe_filename(url, content_type)
                file_path = os.path.join(data_dir, filename)
                
                # Download with size checking
                downloaded_size = 0
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_file_size:
                            await f.close()
                            os.remove(file_path)
                            logger.warning(f"File size exceeded during download")
                            return None
                        await f.write(chunk)
                
                file_size = os.path.getsize(file_path)
                
                return {
                    "title": filename,
                    "file_path": file_path,
                    "content_type": content_type,
                    "file_size": file_size,
                    "download_method": "http",
                    "downloaded_at": datetime.utcnow().isoformat()
                }
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error downloading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return None

class YouTubeDownloadStrategy(DownloadStrategy):
    """YouTube content extraction"""
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Extract YouTube metadata without downloading video"""
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError, ExtractorError
            
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            # Save metadata as JSON
            filename = f"youtube_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
            file_path = os.path.join(data_dir, filename)
            
            import json
            metadata = {
                "title": info.get("title"),
                "description": info.get("description"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "duration": info.get("duration"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "tags": info.get("tags", []),
                "categories": info.get("categories", [])
            }
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            return {
                "title": filename,
                "file_path": file_path,
                "content_type": "application/json",
                "file_size": os.path.getsize(file_path),
                "download_method": "youtube",
                "downloaded_at": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
            
        except (DownloadError, ExtractorError) as e:
            logger.error(f"YouTube extraction error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting YouTube content: {e}")
            return None
