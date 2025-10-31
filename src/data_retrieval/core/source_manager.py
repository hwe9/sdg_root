import os
import csv
import logging
import aiohttp
import asyncio
import inspect
from datetime import datetime
from typing import Set
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from urllib.parse import urlparse

from ..sources.web_source import WebSource
from ..sources.api_source import APISource
from ..sources.feed_source import FeedSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceManager:
        
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
        
        # Cache for processed URLs and content hashes
        self._processed_urls_cache = None
        self._processed_hashes_cache = None
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
    
    async def validate_all_sources(self) -> List[Dict[str, Any]]:
        urls = await self.get_all_sources()
        results = []

        timeout = aiohttp.ClientTimeout(total=10)
        headers = {"User-Agent": "SDG-Pipeline-Bot/2.0 (+https://sdg-pipeline.org/bot)"}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for url in urls:
                status = "unknown"
                reason = ""
                try:
                    # Erst HEAD versuchen, dann Fallback GET mit kleinem Body
                    async with session.head(url, allow_redirects=True) as resp:
                        status = "ok" if resp.status < 400 else f"http_{resp.status}"
                except Exception as e:
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            status = "ok" if resp.status < 400 else f"http_{resp.status}"
                    except Exception as e2:
                        status = "error"
                        reason = str(e2)

                results.append({
                    "url": url,
                    "status": status,
                    "reason": reason
                })

        return results

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
    
    async def get_processed_content_hashes(self) -> Set[str]:
        """Get set of already processed content hashes (for deduplication)"""
        current_time = datetime.utcnow()
        
        # Use cache if recent (within 5 minutes)
        if (self._processed_hashes_cache is not None and 
            self._cache_last_updated is not None and
            (current_time - self._cache_last_updated).seconds < 300):
            return self._processed_hashes_cache
        
        processed_hashes = set()
        
        try:
            if os.path.exists(self.downloaded_urls_file):
                with open(self.downloaded_urls_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('status') == 'success':
                            content_hash = row.get('content_hash', '').strip()
                            if content_hash:
                                processed_hashes.add(content_hash)
                
                logger.info(f"✅ Loaded {len(processed_hashes)} processed content hashes from history")
            
            # Update cache
            self._processed_hashes_cache = processed_hashes
            self._cache_last_updated = current_time
            
            return processed_hashes
            
        except Exception as e:
            logger.error(f"❌ Error loading processed content hashes: {e}")
            return set()
    
    async def mark_url_processed(self, url: str, status: str, filename: str = "", content_hash: str = ""):
        """Mark URL as processed with status and optional content hash"""
        try:
            file_exists = os.path.exists(self.downloaded_urls_file)
            
            with open(self.downloaded_urls_file, 'a', encoding='utf-8', newline='') as f:
                fieldnames = ['url', 'filename', 'timestamp', 'status', 'content_hash']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'url': url,
                    'filename': filename,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': status,
                    'content_hash': content_hash
                })
            
            # Invalidate cache
            self._processed_urls_cache = None
            self._processed_hashes_cache = None
            
            logger.debug(f"✅ Marked URL as {status}: {url} (hash: {content_hash[:16] if content_hash else 'N/A'}...)")
            
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
    async def validate_all_sources(self) -> List[Dict[str, Any]]:
        """
        Lightweight accessibility check for all sources.
        Returns list of {url, reachable: bool, status: str}
        """
        results: List[Dict[str, Any]] = []
        import aiohttp
        import asyncio
        urls = await self.get_all_sources()
        timeout = aiohttp.ClientTimeout(total=15)
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            sem = asyncio.Semaphore(10)
            async def probe(u: str):
                async with sem:
                    try:
                        async with session.head(u, allow_redirects=True) as resp:
                            results.append({"url": u, "reachable": resp.status < 400, "status": str(resp.status)})
                    except Exception as e:
                        results.append({"url": u, "reachable": False, "status": str(e)})
            await asyncio.gather(*[probe(u) for u in urls])
        return results
    
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
        
    async def cleanup(self):
        """Cleanup all source handlers and close sessions"""
        try:
            for handler in self.source_handlers.values():
                if hasattr(handler, "cleanup"):
                    if inspect.iscoroutinefunction(handler.cleanup):
                        await handler.cleanup()
                    else:
                        handler.cleanup()
            logger.info("✅ Source manager cleaned up")
        except Exception as e:
            logger.warning(f"SourceManager cleanup warning: {e}")
    
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
