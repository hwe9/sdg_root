"""
Enhanced Retrieval Engine - Main orchestration component
Coordinates all retrieval activities with improved error handling and monitoring
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import os
from dataclasses import dataclass, asdict

from .source_manager import SourceManager
from .content_validator import ContentValidator
from ..middleware.rate_limiter import RateLimiter
from ..middleware.security_validator import SecurityValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalStats:
    total_sources: int = 0
    processed_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RetrievalEngine:
    """
    Enhanced retrieval engine with comprehensive error handling,
    security validation, and performance monitoring
    """
    
    def __init__(self, source_manager: SourceManager, data_dir: str, processed_file: str):
        self.source_manager = source_manager
        self.data_dir = data_dir
        self.processed_file = processed_file
        
        # Initialize components
        self.content_validator = ContentValidator()
        self.rate_limiter = RateLimiter()
        self.security_validator = SecurityValidator()
        
        # Status tracking
        self.current_stats = RetrievalStats()
        self.is_running = False
        self.last_run_time = None
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    
    async def initialize(self):
        """Initialize engine components"""
        try:
            await self.source_manager.initialize()
            await self.content_validator.initialize()
            logger.info("✅ Retrieval engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval engine: {e}")
            raise
    
    async def run_retrieval(self, 
                          sources: Optional[List[str]] = None,
                          force_refresh: bool = False,
                          max_concurrent: int = 5) -> RetrievalStats:
        """
        Run enhanced retrieval process with improved orchestration
        """
        if self.is_running:
            logger.warning("Retrieval already in progress")
            return self.current_stats
        
        self.is_running = True
        self.current_stats = RetrievalStats(start_time=datetime.utcnow())
        
        try:
            logger.info("🚀 Starting enhanced retrieval process...")
            
            # Get sources to process
            if sources:
                source_urls = set(sources)
            else:
                source_urls = await self.source_manager.get_all_sources()
            
            self.current_stats.total_sources = len(source_urls)
            
            # Filter already processed URLs if not forcing refresh
            if not force_refresh:
                processed_urls = await self.source_manager.get_processed_urls()
                new_urls = source_urls - processed_urls
                self.current_stats.skipped_count = len(source_urls) - len(new_urls)
                source_urls = new_urls
            
            if not source_urls:
                logger.info("🔄 No new URLs to process")
                return self.current_stats
            
            logger.info(f"📥 Processing {len(source_urls)} URLs with {max_concurrent} concurrent workers")
            
            # Process URLs with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [
                self._process_single_url(url, semaphore) 
                for url in source_urls
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Task failed: {result}")
                    self.current_stats.failed_count += 1
                elif result:
                    processed_data.append(result)
                    self.current_stats.success_count += 1
                else:
                    self.current_stats.failed_count += 1
                
                self.current_stats.processed_count += 1
            
            # Save processed data
            await self._save_processed_data(processed_data)
            
            # Signal processing service
            await self._signal_processing_service(processed_data)
            
            self.current_stats.end_time = datetime.utcnow()
            self.last_run_time = self.current_stats.end_time
            
            logger.info(f"✅ Retrieval completed: {self.current_stats.success_count} successful, "
                       f"{self.current_stats.failed_count} failed, {self.current_stats.skipped_count} skipped")
            
            return self.current_stats
            
        except Exception as e:
            logger.error(f"❌ Retrieval process failed: {e}")
            self.current_stats.end_time = datetime.utcnow()
            raise
        finally:
            self.is_running = False
    
    async def _process_single_url(self, url: str, semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
        """Process a single URL with comprehensive validation"""
        async with semaphore:
            try:
                # Security validation
                security_result = await self.security_validator.validate_url(url)
                if not security_result.is_valid:
                    logger.warning(f"🚨 Security validation failed for {url}: {security_result.reason}")
                    await self.source_manager.mark_url_processed(url, "security_blocked")
                    return None
                
                # Rate limiting
                await self.rate_limiter.wait_for_rate_limit(url)
                
                # Download content using appropriate strategy
                download_result = await self.source_manager.download_content(url)
                if not download_result:
                    await self.source_manager.mark_url_processed(url, "download_failed")
                    return None
                
                # Content validation
                validation_result = await self.content_validator.validate_content(download_result)
                if not validation_result.is_valid:
                    logger.warning(f"⚠️ Content validation failed for {url}: {validation_result.reason}")
                    await self.source_manager.mark_url_processed(url, "content_invalid")
                    return None
                
                # Mark as successfully processed
                await self.source_manager.mark_url_processed(url, "success")
                
                # Prepare metadata
                metadata = {
                    "url": url,
                    "title": download_result.get("title", "Unknown"),
                    "file_path": download_result.get("file_path"),
                    "content_type": download_result.get("content_type"),
                    "file_size": download_result.get("file_size"),
                    "source_url": url,
                    "processed_at": datetime.utcnow().isoformat(),
                    "validation_score": validation_result.quality_score,
                    "security_validated": True
                }
                
                # Save individual metadata file
                await self._save_metadata_file(url, metadata)
                
                return metadata
                
            except Exception as e:
                logger.error(f"❌ Error processing {url}: {e}")
                await self.source_manager.mark_url_processed(url, "error")
                return None
    
    async def _save_processed_data(self, processed_data: List[Dict[str, Any]]):
        """Save processed data to JSON file"""
        try:
            with open(self.processed_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Saved {len(processed_data)} processed items to {self.processed_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save processed data: {e}")
    
    async def _save_metadata_file(self, url: str, metadata: Dict[str, Any]):
        """Save individual metadata file"""
        try:
            # Generate safe filename
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"metadata_{url_hash}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"✅ Saved metadata for {url}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata for {url}: {e}")
    
    async def _signal_processing_service(self, processed_data: List[Dict[str, Any]]):
        """Signal processing service about new data"""
        try:
            # Create signal file for processing service
            signal_data = {
                "processed_file": self.processed_file,
                "item_count": len(processed_data),
                "processed_at": datetime.utcnow().isoformat(),
                "service": "data_retrieval"
            }
            
            signal_file = os.path.join(os.path.dirname(self.processed_file), "processing_signal.json")
            with open(signal_file, "w", encoding="utf-8") as f:
                json.dump(signal_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🔔 Signaled processing service: {len(processed_data)} items ready")
        except Exception as e:
            logger.error(f"❌ Failed to signal processing service: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current retrieval status"""
        return {
            "is_running": self.is_running,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "current_stats": self.current_stats.to_dict(),
            "data_directory": self.data_dir,
            "processed_file": self.processed_file
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for retrieval engine"""
        try:
            return {
                "status": "healthy",
                "is_running": self.is_running,
                "last_successful_run": self.last_run_time.isoformat() if self.last_run_time else None,
                "components": {
                    "source_manager": self.source_manager.health_check(),
                    "content_validator": self.content_validator.health_check(),
                    "rate_limiter": self.rate_limiter.health_check(),
                    "security_validator": self.security_validator.health_check()
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
