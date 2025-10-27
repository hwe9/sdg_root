"""
Web Source Handler
Handles HTTP/HTTPS downloads with proper error handling
"""
import logging
from typing import Dict
from typing import Any
from typing import Optional

from ..core.download_strategies import HTTPDownloadStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSource:
    """Handler for general web content downloads"""
    
    def __init__(self):
        self.strategy = HTTPDownloadStrategy()
    
    async def initialize(self):
        """Initialize web source handler"""
        await self.strategy.initialize()
        logger.info("✅ Web source handler initialized")
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Download web content"""
        return await self.strategy.download(url, data_dir)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.strategy.cleanup()
