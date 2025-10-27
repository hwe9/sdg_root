"""
Base extractor class for all content extraction sources
"""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Any
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from urllib.parse import urljoin
from urllib.parse import urlparse
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Standard format for extracted content"""
    title: str
    content: str
    summary: Optional[str] = None
    url: str = ""
    source_type: str = ""
    language: str = "en"
    region: str = ""
    extracted_at: datetime = None
    metadata: Dict[str, Any] = None
    quality_score: float = 0.0
    sdg_relevance: List[int] = None
    
    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.sdg_relevance is None:
            self.sdg_relevance = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "url": self.url,
            "source_type": self.source_type,
            "language": self.language,
            "region": self.region,
            "extracted_at": self.extracted_at.isoformat(),
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "sdg_relevance": self.sdg_relevance
        }
    
    def content_hash(self) -> str:
        """Generate content hash for deduplication"""
        content_string = f"{self.title}{self.content}{self.url}"
        return hashlib.md5(content_string.encode()).hexdigest()

class BaseExtractor(ABC):
    """
    Abstract base class for all content extractors
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout = config.get("timeout", 30)
        self.user_agent = config.get("user_agent", 
            "SDG-Pipeline-Bot/1.0 (+https://sdg-pipeline.org/bot)")
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract content from source"""
        pass
    
    @abstractmethod
    def validate_source(self, source_url: str) -> bool:
        """Validate if source URL is supported"""
        pass
    
    async def fetch_with_retry(self, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Fetch URL with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.get(url, **kwargs) as response:
                    if response.status == 200:
                        return response
                    elif response.status in [429, 503, 504]:  # Rate limiting or server errors
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited on {url}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
        
        return None
    
    def estimate_quality_score(self, content: ExtractedContent) -> float:
        """Estimate content quality score"""
        score = 0.0
        
        # Title quality (0-0.2)
        if content.title and len(content.title.strip()) > 10:
            score += 0.2
        
        # Content length (0-0.3)
        content_length = len(content.content.strip())
        if content_length > 500:
            score += 0.3
        elif content_length > 200:
            score += 0.15
        
        # Has URL (0-0.1)
        if content.url and content.url.startswith(('http://', 'https://')):
            score += 0.1
        
        # Has metadata (0-0.2)
        if content.metadata and len(content.metadata) > 2:
            score += 0.2
        
        # Language detection (0-0.1)
        if content.language and content.language != "unknown":
            score += 0.1
        
        # SDG relevance (0-0.1)
        if content.sdg_relevance and len(content.sdg_relevance) > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    async def process_batch(self, urls: List[str], **kwargs) -> List[ExtractedContent]:
        """Process multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(self.config.get("concurrent_requests", 5))
        
        async def extract_single(url: str) -> List[ExtractedContent]:
            async with semaphore:
                try:
                    return await self.extract(url, **kwargs)
                except Exception as e:
                    logger.error(f"Error extracting {url}: {e}")
                    return []
        
        tasks = [extract_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter exceptions
        extracted_content = []
        for result in results:
            if isinstance(result, list):
                extracted_content.extend(result)
        
        return extracted_content
