"""
Web Source Handler
Handles HTTP/HTTPS downloads with proper error handling
"""
import logging
from typing import Dict
from typing import Any
from typing import Optional

from ..core.download_strategies import HTTPDownloadStrategy, YouTubeDownloadStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSource:
    """Handler for general web content downloads, including YouTube"""
    
    def __init__(self):
        self.http_strategy = HTTPDownloadStrategy()
        self.youtube_strategy = YouTubeDownloadStrategy()
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        return any(domain in url.lower() for domain in youtube_domains)
    
    async def initialize(self):
        """Initialize web source handler"""
        await self.http_strategy.initialize()
        await self.youtube_strategy.initialize()
        logger.info("✅ Web source handler initialized")
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Download web content, routing YouTube URLs to YouTube strategy"""
        if self._is_youtube_url(url):
            return await self.youtube_strategy.download(url, data_dir)
        else:
            return await self.http_strategy.download(url, data_dir)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_strategy.cleanup()
        await self.youtube_strategy.cleanup()
