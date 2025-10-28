# src/content_extraction/main.py

import os
import logging
import asyncio
import time
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import BackgroundTasks
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import field_validator
import httpx

# SDG dependency manager
from ..core.dependency_manager import (
    dependency_manager,
    setup_sdg_dependencies,
    get_dependency_status,
    ServiceStatus,
)
from ..core.logging_config import get_logger

# Configure logging
logger = get_logger("content_extraction")
readiness = {"ready": False}
extractors = {}

# Extractors
try:
    from .extractors.gemini_extractor import GeminiExtractor
    from .extractors.web_extractor import WebExtractor
    from .extractors.newsletter_extractor import NewsletterExtractor
    from .extractors.rss_extractor import RSSExtractor
except ImportError as e:
    logger.error(f"failed to import extractors: {e}")

    class DummyExtractor:
        def __init__(self, config): ...
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): ...
        async def extract(self, *args, **kwargs): return []
        async def process_batch(self, *args, **kwargs): return []
        def validate_source(self, *args, **kwargs): return False

    GeminiExtractor = DummyExtractor  # fallback
    WebExtractor = DummyExtractor     # fallback
    NewsletterExtractor = DummyExtractor  # fallback
    RSSExtractor = DummyExtractor     # fallback

# Global extractor registry
extractors: Dict[str, Any] = {}

class ExtractionRequest(BaseModel):
    url: HttpUrl = Field(..., description="Source URL to extract content from")
    source_type: Optional[str] = Field(None, description="Source type hint")
    language: Optional[str] = Field("en", description="Expected content language")
    region: Optional[str] = Field(None, description="Expected content region")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: HttpUrl):
        url_str = str(v)
        dangerous_protocols = [
            'file://', 'ftp://', 'javascript:', 'data:', 'gopher://',
            'ldap://', 'dict://', 'sftp://', 'tftp://', 'telnet://'
        ]
        if any(url_str.lower().startswith(proto) for proto in dangerous_protocols):
            raise ValueError(f'Unsafe URL protocol: {url_str[:20]}')
        import re
        ip_pattern = r'https?://(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        if re.match(ip_pattern, url_str.lower()):
            raise ValueError('Direct IP addresses not allowed')
        localhost_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        if any(pattern in url_str.lower() for pattern in localhost_patterns):
            raise ValueError('Localhost URLs not allowed')
        return v

class BatchExtractionRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="List of URLs to extract", max_items=20)
    source_type: Optional[str] = Field(None, description="Source type for all URLs")
    language: Optional[str] = Field("en", description="Expected content language")
    region: Optional[str] = Field(None, description="Expected content region")

class GeminiAnalysisRequest(BaseModel):
    content: str = Field(..., description="Content to analyze with Gemini", min_length=100)
    source_url: Optional[HttpUrl] = Field(None, description="Original source URL")
    language: Optional[str] = Field("en", description="Content language")

class ExtractionResponse(BaseModel):
    success: bool
    extracted_count: int
    content: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float

async def initialize_extractors():
    """Initialize extraction services"""
    config = {
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "timeout": 30,
        "concurrent_requests": 5,
        "max_entries_per_feed": 50,
        "user_agent": "SDG-Pipeline-ContentExtractor/1.0",
    }
    try:
        extractors['web'] = WebExtractor(config)
        extractors['newsletter'] = NewsletterExtractor(config)
        extractors['rss'] = RSSExtractor(config)
        extractors['gemini'] = GeminiExtractor(config)
        logger.info("Content extractors initialized")
    except Exception as e:
        logger.error(f"Failed to initialize extractors: {e}")
        # Fallback to dummies for all types
        for name in ('web', 'newsletter', 'rss', 'gemini'):
            extractors[name] = DummyExtractor(config)

def _norm(name: str) -> str:
    return name.replace("_", "").lower()

readiness = {"ready": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Content Extraction Service...")
    setup_sdg_dependencies()

    # Mark non-critical deps optional
    for svc in ("auth", "api", "data_processing", "vectorization", "data_retrieval", "weaviate", "weaviate_transformer"):
        key = _norm(svc)
        if key in getattr(dependency_manager, "services", {}):
            dependency_manager.services[key].required = False
            logger.info(f"Marked {key} as optional for content_extraction")

    # IMPORTANT: do NOT pop self from services; keep it registered to avoid KeyError in get_service_status
    # If desired, also make self optional (defensive)
    for self_key in ("content_extraction", "contentextraction"):
        key = _norm(self_key)
        if key in getattr(dependency_manager, "services", {}):
            dependency_manager.services[key].required = False

    # Start dependency manager in background
    asyncio.create_task(dependency_manager.start_all_services())

    # Wait only for database to become healthy
    db_key = _norm("database")
    t0 = time.time()
    timeout = 120.0
    while time.time() - t0 < timeout:
        status = dependency_manager.service_status.get(db_key)
        if status == ServiceStatus.HEALTHY:
            break
        await asyncio.sleep(1.0)
    else:
        raise TimeoutError("Database did not become healthy in time for content_extraction startup")

    # Initialize extractors
    await initialize_extractors()
    readiness["ready"] = True
    logger.info("✅ Content extraction service initialized")

    yield

    # --- shutdown ---
    logger.info("🔄 Shutting down Content Extraction Service...")
    try:
        for extractor in extractors.values():
            if hasattr(extractor, 'session') and getattr(extractor, 'session', None):
                await extractor.session.close()
    except Exception:
        logger.exception("Error during extractor shutdown")


# FastAPI app with lifespan
app = FastAPI(
    title="SDG Content Extraction Service",
    description="Microservice for extracting and analyzing content from multiple sources",
    version="2.0.0",
    lifespan=lifespan,
)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    try:
        dependency_status = await get_dependency_status()
    except Exception as e:
        # Return a stable response even if DM raises (e.g., transient inconsistencies)
        dependency_status = {"error": str(e), "services": {}}
    return {
        "status": "ok" if readiness.get("ready") else "starting",
        "ready": readiness.get("ready", False),
        "dependencies": dependency_status,
    }

# Single URL extraction
@app.post("/extract", response_model=ExtractionResponse)
async def extract_content(request: ExtractionRequest):
    start_time = datetime.now()
    try:
        extractor_type = await determine_extractor_type(str(request.url), request.source_type)
        extractor = extractors.get(extractor_type)
        if not extractor:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {extractor_type}")
        async with extractor:
            extracted_content = await extractor.extract(
                str(request.url), language=request.language, region=request.region
            )
        processing_time = (datetime.now() - start_time).total_seconds()
        return ExtractionResponse(
            success=True,
            extracted_count=len(extracted_content),
            content=[item.to_dict() for item in extracted_content],
            metadata={
                "source_url": str(request.url),
                "extractor_used": extractor_type,
                "processing_time": processing_time,
            },
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error extracting from {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch extraction
@app.post("/extract/batch", response_model=ExtractionResponse)
async def extract_batch_content(request: BatchExtractionRequest):
    start_time = datetime.now()
    try:
        all_content: List[Any] = []
        extractor_usage: Dict[str, int] = {}
        url_groups: Dict[str, List[str]] = {}
        for url in request.urls:
            extractor_type = await determine_extractor_type(str(url), request.source_type)
            url_groups.setdefault(extractor_type, []).append(str(url))
        for extractor_type, urls in url_groups.items():
            extractor = extractors.get(extractor_type)
            if not extractor:
                logger.warning(f"Skipping unsupported extractor type: {extractor_type}")
                continue
            async with extractor:
                batch_content = await extractor.process_batch(
                    urls, language=request.language, region=request.region
                )
                all_content.extend(batch_content)
                extractor_usage[extractor_type] = len(urls)
        processing_time = (datetime.now() - start_time).total_seconds()
        return ExtractionResponse(
            success=True,
            extracted_count=len(all_content),
            content=[item.to_dict() for item in all_content],
            metadata={
                "total_urls": len(request.urls),
                "extractor_usage": extractor_usage,
                "processing_time": processing_time,
            },
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error in batch extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RSS extraction
@app.post("/extract/rss", response_model=ExtractionResponse)
async def extract_rss_feeds(
    feed_urls: List[HttpUrl],
    max_entries: int = Query(50, description="Maximum entries per feed"),
    language: Optional[str] = Query("en", description="Expected language"),
    region: Optional[str] = Query(None, description="Expected region"),
):
    start_time = datetime.now()
    try:
        rss_extractor = extractors['rss']
        # If extractor supports it, adjust per-request parameter
        if hasattr(rss_extractor, "max_entries"):
            setattr(rss_extractor, "max_entries", max_entries)
        async with rss_extractor:
            all_content = await rss_extractor.extract_multiple_feeds(
                [str(url) for url in feed_urls], language=language, region=region
            )
        processing_time = (datetime.now() - start_time).total_seconds()
        return ExtractionResponse(
            success=True,
            extracted_count=len(all_content),
            content=[item.to_dict() for item in all_content],
            metadata={
                "feed_count": len(feed_urls),
                "max_entries_per_feed": max_entries,
                "processing_time": processing_time,
            },
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error extracting RSS feeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gemini analysis
@app.post("/analyze/gemini", response_model=ExtractionResponse)
async def analyze_with_gemini(request: GeminiAnalysisRequest):
    start_time = datetime.now()
    try:
        gemini_extractor = extractors['gemini']
        async with gemini_extractor:
            analyzed_content = await gemini_extractor.extract(
                request.content,
                source_url=str(request.source_url) if request.source_url else "",
                language=request.language,
            )
        processing_time = (datetime.now() - start_time).total_seconds()
        return ExtractionResponse(
            success=True,
            extracted_count=len(analyzed_content),
            content=[item.to_dict() for item in analyzed_content],
            metadata={
                "analysis_type": "gemini_2.5",
                "content_length": len(request.content),
                "processing_time": processing_time,
            },
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility
@app.get("/extractors")
async def get_available_extractors():
    return {
        "extractors": {
            name: {
                "type": type(extractor).__name__,
                "description": (extractor.__class__.__doc__ or "").strip(),
            }
            for name, extractor in extractors.items()
        }
    }

@app.post("/validate-url")
async def validate_url(url: HttpUrl, source_type: Optional[str] = None):
    try:
        extractor_type = await determine_extractor_type(str(url), source_type)
        extractor = extractors.get(extractor_type)
        if not extractor:
            return {"valid": False, "reason": f"No extractor available for type: {extractor_type}"}
        is_valid = extractor.validate_source(str(url))
        return {"valid": is_valid, "extractor_type": extractor_type, "supported": True}
    except Exception as e:
        return {"valid": False, "reason": str(e)}

# Pipeline forwarding
async def forward_to_processing_service(content_items: List[Dict[str, Any]]):
    try:
        processing_url = "http://data_processing_service:8001/process-content"
        async with httpx.AsyncClient() as client:
            resp = await client.post(processing_url, json={"content_items": content_items}, timeout=60)
            if resp.status_code == 200:
                logger.info(f"Successfully forwarded {len(content_items)} items to processing service")
            else:
                logger.error(f"Error forwarding to processing service: {resp.status_code}")
    except Exception as e:
        logger.error(f"Error forwarding content to processing service: {e}")

# Helper
async def determine_extractor_type(url: str, hint: Optional[str] = None) -> str:
    url_lower = url.lower()
    if hint and hint in extractors:
        return hint
    if any(p in url_lower for p in ['/rss', '/feed', '.rss', '.xml']):
        return 'rss'
    if any(p in url_lower for p in ['newsletter', 'digest', 'bulletin']):
        return 'newsletter'
    if url_lower.startswith(('http://', 'https://')):
        return 'web'
    return 'gemini'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True, log_level="info")
