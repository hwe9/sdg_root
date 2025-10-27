import asyncio
import logging
from datetime import datetime
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
import json
import os
from dataclasses import dataclass
from dataclasses import asdict

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
    def __init__(self, source_manager: SourceManager, data_dir: str, processed_file: str):
        self.source_manager = source_manager
        self.data_dir = data_dir
        self.processed_file = processed_file
        self.content_validator = ContentValidator()

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _env_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            v = v.strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            return default

        self.rate_limiter = RateLimiter(
            _env_float("RATE_LIMIT_GLOBAL_RPS", 10.0),           
            _env_float("RATE_LIMIT_PER_DOMAIN_RPS", 1.5),        
            _env_bool("ROBOTS_RESPECT_CRAWL_DELAY", True),       
            _env_float("ROBOTS_DEFAULT_DELAY_SEC", 1.0),         
            _env_int("RATE_LIMIT_JITTER_MS", 200),               
        )
        self.security_validator = SecurityValidator()
        self.current_stats = RetrievalStats()
        self.is_running = False
        self.last_run_time = None

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)

        logger.info("Retrieval engine initialized successfully")
    
    async def initialize(self):
        try:
            await self.source_manager.initialize()
            await self.content_validator.initialize()
            await self.rate_limiter.initialize()
            logger.info("✅ Retrieval engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval engine: {e}")
            raise
    
    async def run_retrieval(
        self,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
        max_concurrent: int = 5,
        filter_region: Optional[str] = None,  
        sdg_goals: Optional[List[int]] = None 
    ) -> RetrievalStats:
        if self.is_running:
            logger.warning("Retrieval already in progress")
            return self.current_stats

        self.is_running = True
        self.current_stats = RetrievalStats()

        try:
            logger.info("🚀 Starting enhanced retrieval process...")

            # Quellen ermitteln
            if sources:
                source_urls = set(sources)
            else:
                source_urls = await self.source_manager.get_all_sources()

            self.current_stats.total_sources = len(source_urls)

            # Bereits verarbeitete URLs filtern (falls nicht force_refresh)
            if not force_refresh:
                processed_urls = await self.source_manager.get_processed_urls()
                new_urls = source_urls - processed_urls
                self.current_stats.skipped_count = len(source_urls) - len(new_urls)
                source_urls = new_urls

            # Optionale Filter nur protokollieren (keine URL-Metadaten verfügbar)
            if filter_region:
                logger.info(f"Region filter requested: {filter_region} (no-op at URL stage)")
            if sdg_goals:
                logger.info(f"SDG filter requested: {sdg_goals} (no-op at URL stage)")

            if not source_urls:
                logger.info("🔄 No new URLs to process")
                return self.current_stats

            logger.info(f"📥 Processing {len(source_urls)} URLs with {max_concurrent} concurrent workers")

            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [self._process_single_url(url, semaphore) for url in source_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_data = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Task failed: {result}")
                    self.current_stats.failed_count += 1
                elif result:
                    processed_data.append(result)
                    self.current_stats.success_count += 1
                else:
                    self.current_stats.failed_count += 1

                self.current_stats.processed_count += 1

            await self._save_processed_data(processed_data)
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
        try:
            with open(self.processed_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Saved {len(processed_data)} processed items to {self.processed_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save processed data: {e}")
    
    async def _save_metadata_file(self, url: str, metadata: Dict[str, Any]):
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

    def get_active_worker_count(self) -> int:
        # Could be refined to reflect active asyncio tasks
        return 0
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "current_stats": self.current_stats.to_dict(),
            "data_directory": self.data_dir,
            "processed_file": self.processed_file
        }
    
    def health_check(self) -> Dict[str, Any]:
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
        
    async def cleanup(self):
        """Cleanup all underlying source handlers"""
        try:
            if self.source_manager:
                await self.source_manager.cleanup()
            logger.info("✅ Retrieval engine cleaned up")
        except Exception as e:
            logger.warning(f"Retrieval engine cleanup warning: {e}")
