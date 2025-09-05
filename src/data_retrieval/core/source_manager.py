"""
Dynamic Source Management
Handles different source types and maintains download history
"""
import os
import csv
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional, List
from urllib.parse import urlparse

from ..sources.web_source import WebSource
from ..sources.api_source import APISource
from ..sources.feed_source import FeedSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceManager:
    """
    Manages different types of data sources and coordinates downloads
    """
    
    def __init__(self, sources_file: str, data_dir: str):
        self.sources_file = sources_file
        self.data_dir = data_dir
        self.downloaded_urls_file = os.path.join(data_dir, "downloaded_urls.csv")
        
        # Initialize source handlers
        self.source_handlers = {
            'web': WebSource(),
            'api': APISource(),
            'feed': FeedSource()
        }
        
        # Cache for processed URLs
        self._processed_urls_cache = None
        self._cache_last_updated = None
    
    async def initialize(self):
        """Initialize source manager and handlers"""
        try:
            # Initialize all source handlers
            for handler in self.source_handlers.values():
                await handler.initialize()
            
            # Ensure data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            logger.info("✅ Source manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize source manager: {e}")
            raise
    
    async def get_all_sources(self) -> Set[str]:
        """Load all configured sources"""
        all_urls = set()
        
        try:
            if os.path.exists(self.sources_file):
                with open(self.sources_file, "r", encoding="utf-8") as f:
                    for line in f:
                        url = line.strip()
                        if url and not url.startswith('#'):  # Skip comments
                            all_urls.add(url)
            
            logger.info(f"📋 Loaded {len(all_urls)} URLs from sources file")
            return all_urls
            
        except Exception as e:
            logger.error(f"❌ Error loading sources: {e}")
            return set()
    
    async def get_processed_urls(self) -> Set[str]:
        """Get set of already processed URLs with caching"""
        current_time = datetime.utcnow()
        
        # Use cache if recent (within 5 minutes)
        if (self._processed_urls_cache is not None and 
            self._cache_last_updated is not None and
            (current_time - self._cache_last_updated).seconds < 300):
            return self._processed_urls_cache
        
        processed_urls = set()
        
        try:
            if os.path.exists(self.downloaded_urls_file):
                with open(self.downloaded_urls_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('status') == 'success':
                            processed_urls.add(row['url'])
                
                logger.info(f"✅ Loaded {len(processed_urls)} processed URLs from history")
            
            # Update cache
            self._processed_urls_cache = processed_urls
            self._cache_last_updated = current_time
            
            return processed_urls
            
        except Exception as e:
            logger.error(f"❌ Error loading processed URLs: {e}")
            return set()
    
    async def mark_url_processed(self, url: str, status: str, filename: str = ""):
        """Mark URL as processed with status"""
        try:
            file_exists = os.path.exists(self.downloaded_urls_file)
            
            with open(self.downloaded_urls_file, 'a', encoding='utf-8', newline='') as f:
                fieldnames = ['url', 'filename', 'timestamp', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'url': url,
                    'filename': filename,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': status
                })
            
            # Invalidate cache
            self._processed_urls_cache = None
            
            logger.debug(f"✅ Marked URL as {status}: {url}")
            
        except Exception as e:
            logger.error(f"❌ Error marking URL processed: {e}")
    
    async def download_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Download content using appropriate source handler"""
        try:
            # Determine source type
            source_type = self._determine_source_type(url)
            handler = self.source_handlers.get(source_type, self.source_handlers['web'])
            
            # Download using appropriate handler
            result = await handler.download(url, self.data_dir)
            
            if result:
                logger.info(f"✅ Successfully downloaded: {url}")
                return result
            else:
                logger.warning(f"⚠️ Download failed: {url}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading {url}: {e}")
            return None
    
    def _determine_source_type(self, url: str) -> str:
        """Determine the appropriate source handler for URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # API endpoints
        if ('api.' in domain or 
            '/api/' in path or 
            'worldbank.org' in domain or
            'un.org' in domain):
            return 'api'
        
        # Feed URLs
        if (path.endswith(('.rss', '.xml', '.atom')) or
            'feed' in path or
            'rss' in path):
            return 'feed'
        
        # Default to web scraping
        return 'web'
    
    def get_source_summary(self) -> Dict[str, Any]:
        """Get summary of configured sources"""
        try:
            all_sources = set()
            
            if os.path.exists(self.sources_file):
                with open(self.sources_file, "r", encoding="utf-8") as f:
                    for line in f:
                        url = line.strip()
                        if url and not url.startswith('#'):
                            all_sources.add(url)
            
            # Categorize sources
            source_types = {'web': 0, 'api': 0, 'feed': 0}
            for url in all_sources:
                source_type = self._determine_source_type(url)
                source_types[source_type] += 1
            
            return {
                "total_sources": len(all_sources),
                "source_types": source_types,
                "sources_file": self.sources_file,
                "data_directory": self.data_dir
            }
            
        except Exception as e:
            logger.error(f"Error getting source summary: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for source manager"""
        try:
            return {
                "status": "healthy",
                "sources_file_exists": os.path.exists(self.sources_file),
                "data_directory_exists": os.path.exists(self.data_dir),
                "handlers_initialized": len(self.source_handlers)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
