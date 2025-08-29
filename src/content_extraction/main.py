"""
Content Extraction Service - FastAPI Application
Handles extraction from multiple sources: Gemini, web pages, newsletters, RSS feeds
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import httpx

try:
    from extractors.gemini_extractor import GeminiExtractor
    from extractors.web_extractor import WebExtractor
    from extractors.newsletter_extractor import NewsletterExtractor
    from extractors.rss_extractor import RSSExtractor

except ImportError as e:
    logging.error(f"failed to import ectractors: {e}")

    class DummyExtractor:
        def __init__(self, config): pass
        async def extract(self, *args, **kwargs): return []
    
    GeminiExtractor = DummyExtractor
    WebExtractor = DummyExtractor
    NewsletterExtractor = DummyExtractor
    RSSExtractor = DummyExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ExtractionRequest(BaseModel):
    url: HttpUrl = Field(..., description="Source URL to extract content from")
    source_type: Optional[str] = Field(None, description="Source type hint")
    language: Optional[str] = Field("en", description="Expected content language")
    region: Optional[str] = Field(None, description="Expected content region")
    
    @validator('url')
    def validate_url(cls, v):
        # Enhanced URL validation
        url_str = str(v)
        
        # Block dangerous protocols
        dangerous_protocols = ['file://', 'ftp://', 'javascript:', 'data:']
        if any(url_str.lower().startswith(proto) for proto in dangerous_protocols):
            raise ValueError('Unsafe URL protocol')
        
        # Block localhost and private IPs
        import re
        if re.search(r'(localhost|127\.0\.0\.1|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)', url_str):
            raise ValueError('Local URLs not allowed')
            
        return v

# Global extractors with error handling
extractors = {}

async def initialize_extractors():
    """Initialize extraction services with error handling"""
    config = {
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "timeout": 30,
        "concurrent_requests": 5,
        "user_agent": "SDG-Pipeline-ContentExtractor/1.0"
    }
    
    try:
        extractors['web'] = WebExtractor(config)
        extractors['newsletter'] = NewsletterExtractor(config)
        extractors['rss'] = RSSExtractor(config)
        extractors['gemini'] = GeminiExtractor(config)
        logger.info("Content extractors initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize extractors: {e}")
        # Initialize with dummy extractors as fallback
        for extractor_type in ['web', 'newsletter', 'rss', 'gemini']:
            extractors[extractor_type] = DummyExtractor(config)
    
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

# Global extractors
extractors = {}

async def initialize_extractors():
    """Initialize extraction services"""
    config = {
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "timeout": 30,
        "concurrent_requests": 5,
        "max_entries_per_feed": 50,
        "user_agent": "SDG-Pipeline-ContentExtractor/1.0"
    }
    
    extractors['web'] = WebExtractor(config)
    extractors['newsletter'] = NewsletterExtractor(config)
    extractors['rss'] = RSSExtractor(config)
    extractors['gemini'] = GeminiExtractor(config)
    
    logger.info("Content extractors initialized")

# FastAPI app
app = FastAPI(
    title="SDG Content Extraction Service",
    description="Microservice for extracting and analyzing content from multiple sources",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_extractors()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    for extractor in extractors.values():
        if hasattr(extractor, 'session') and extractor.session:
            await extractor.session.close()

# Health check
@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "SDG Content Extraction Service",
        "version": "1.0.0",
        "extractors_loaded": len(extractors),
        "available_extractors": list(extractors.keys())
    }

# Single URL extraction
@app.post("/extract", response_model=ExtractionResponse)
async def extract_content(request: ExtractionRequest):
    """Extract content from a single URL"""
    start_time = datetime.now()
    
    try:
        # Determine extractor type
        extractor_type = await determine_extractor_type(str(request.url), request.source_type)
        extractor = extractors.get(extractor_type)
        
        if not extractor:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {extractor_type}")
        
        # Extract content
        async with extractor:
            extracted_content = await extractor.extract(
                str(request.url),
                language=request.language,
                region=request.region
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResponse(
            success=True,
            extracted_count=len(extracted_content),
            content=[item.to_dict() for item in extracted_content],
            metadata={
                "source_url": str(request.url),
                "extractor_used": extractor_type,
                "processing_time": processing_time
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting from {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch extraction
@app.post("/extract/batch", response_model=ExtractionResponse)
async def extract_batch_content(request: BatchExtractionRequest):
    """Extract content from multiple URLs"""
    start_time = datetime.now()
    
    try:
        all_content = []
        extractor_usage = {}
        
        # Group URLs by extractor type
        url_groups = {}
        for url in request.urls:
            extractor_type = await determine_extractor_type(str(url), request.source_type)
            url_groups.setdefault(extractor_type, []).append(str(url))
        
        # Process each group with appropriate extractor
        for extractor_type, urls in url_groups.items():
            extractor = extractors.get(extractor_type)
            if not extractor:
                logger.warning(f"Skipping unsupported extractor type: {extractor_type}")
                continue
            
            async with extractor:
                batch_content = await extractor.process_batch(
                    urls,
                    language=request.language,
                    region=request.region
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
                "processing_time": processing_time
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RSS feed extraction
@app.post("/extract/rss")
async def extract_rss_feeds(
    feed_urls: List[HttpUrl],
    max_entries: int = Query(50, description="Maximum entries per feed"),
    language: Optional[str] = Query("en", description="Expected language"),
    region: Optional[str] = Query(None, description="Expected region")
):
    """Extract content from RSS feeds"""
    start_time = datetime.now()
    
    try:
        rss_extractor = extractors['rss']
        
        # Update config for this request
        rss_extractor.max_entries = max_entries
        
        async with rss_extractor:
            all_content = await rss_extractor.extract_multiple_feeds(
                [str(url) for url in feed_urls],
                language=language,
                region=region
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResponse(
            success=True,
            extracted_count=len(all_content),
            content=[item.to_dict() for item in all_content],
            metadata={
                "feed_count": len(feed_urls),
                "max_entries_per_feed": max_entries,
                "processing_time": processing_time
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting RSS feeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gemini analysis
@app.post("/analyze/gemini")
async def analyze_with_gemini(request: GeminiAnalysisRequest):
    """Analyze content using Gemini 2.5"""
    start_time = datetime.now()
    
    try:
        gemini_extractor = extractors['gemini']
        
        async with gemini_extractor:
            analyzed_content = await gemini_extractor.extract(
                request.content,
                source_url=str(request.source_url) if request.source_url else "",
                language=request.language
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResponse(
            success=True,
            extracted_count=len(analyzed_content),
            content=[item.to_dict() for item in analyzed_content],
            metadata={
                "analysis_type": "gemini_2.5",
                "content_length": len(request.content),
                "processing_time": processing_time
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/extractors")
async def get_available_extractors():
    """Get information about available extractors"""
    return {
        "extractors": {
            name: {
                "type": type(extractor).__name__,
                "description": extractor.__class__.__doc__.strip() if extractor.__class__.__doc__ else ""
            }
            for name, extractor in extractors.items()
        }
    }

@app.post("/validate-url")
async def validate_url(url: HttpUrl, source_type: Optional[str] = None):
    """Validate if URL can be processed by available extractors"""
    try:
        extractor_type = await determine_extractor_type(str(url), source_type)
        extractor = extractors.get(extractor_type)
        
        if not extractor:
            return {
                "valid": False,
                "reason": f"No extractor available for type: {extractor_type}"
            }
        
        is_valid = extractor.validate_source(str(url))
        
        return {
            "valid": is_valid,
            "extractor_type": extractor_type,
            "supported": True
        }
        
    except Exception as e:
        return {
            "valid": False,
            "reason": str(e)
        }

# Integration with existing SDG pipeline
@app.post("/extract-and-forward")
async def extract_and_forward_to_pipeline(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks
):
    """Extract content and forward to data processing service"""
    try:
        # Extract content
        extraction_result = await extract_content(request)
        
        if extraction_result.success:
            # Forward to data processing service in background
            background_tasks.add_task(
                forward_to_processing_service,
                extraction_result.content
            )
        
        return {
            "extraction_success": extraction_result.success,
            "extracted_count": extraction_result.extracted_count,
            "forwarded_to_processing": True
        }
        
    except Exception as e:
        logger.error(f"Error in extract-and-forward: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def determine_extractor_type(url: str, hint: Optional[str] = None) -> str:
    """Determine which extractor to use for a URL"""
    url_lower = url.lower()
    
    # Use hint if provided and valid
    if hint and hint in extractors:
        return hint
    
    # Auto-detect based on URL patterns
    if any(pattern in url_lower for pattern in ['/rss', '/feed', '.rss', '.xml']):
        return 'rss'
    elif any(pattern in url_lower for pattern in ['newsletter', 'digest', 'bulletin']):
        return 'newsletter'
    elif url_lower.startswith(('http://', 'https://')):
        return 'web'
    else:
        return 'gemini'  # Default for content analysis

async def forward_to_processing_service(content_items: List[Dict[str, Any]]):
    """Forward extracted content to data processing service"""
    try:
        processing_url = "http://data_processing_service:8001/process-content"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                processing_url,
                json={"content_items": content_items},
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully forwarded {len(content_items)} items to processing service")
            else:
                logger.error(f"Error forwarding to processing service: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error forwarding content to processing service: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )
