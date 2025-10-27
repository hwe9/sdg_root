# src/vectorization/main.py
import os
import logging
import sys
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

# Add parent directory to path for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field
import numpy as np
import uvicorn

# Core imports with error handling
try:
    from core.dependency_manager import dependency_manager
    from core.dependency_manager import wait_for_dependencies
    from core.dependency_manager import get_dependency_status
    from core.dependency_manager import setup_sdg_dependencies
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False
logger = get_logger("vectorization")
    logger.warning("Dependency manager not available, falling back to direct initialization")

from .embedding_models import EmbeddingManager
from .vector_db_client import VectorDBClient
from .vector_db_client import get_vector_client
from .similarity_search import SimilaritySearch
from .similarity_search import SDGRecommendationEngine
from ..core.logging_config import get_logger

# Import centralized logging
try:
    from core.logging_config import get_logger
    logger = get_logger("vectorization")
except ImportError:
    # Fallback for backward compatibility
logger = get_logger("vectorization")

# Global service instances
embedding_manager: Optional[EmbeddingManager] = None
vector_client: Optional[VectorDBClient] = None
similarity_search: Optional[SimilaritySearch] = None
recommendation_engine: Optional[SDGRecommendationEngine] = None

# Pydantic models (keeping existing ones)
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    model_name: Optional[str] = Field("sentence_transformer", description="Embedding model to use")
    normalize: bool = Field(True, description="Normalize embeddings")

class DocumentRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")  
    summary: Optional[str] = Field(None, description="Document summary")
    sdg_goals: List[int] = Field(..., description="Related SDG goals")
    region: str = Field(..., description="Geographic region")
    language: str = Field("en", description="Content language")
    source_url: Optional[str] = Field(None, description="Source URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: str = Field("general", description="Search type: general, high_quality, recent, comprehensive")
    language: Optional[str] = Field("en", description="Content language")
    region: Optional[str] = Field(None, description="Geographic region filter")
    sdg_goals: Optional[List[int]] = Field(None, description="SDG goals filter")
    limit: int = Field(10, description="Maximum results", ge=1, le=100)

class RecommendationRequest(BaseModel):
    user_interests: List[int] = Field(..., description="User's SDG interests (1-17)")
    region: Optional[str] = Field(None, description="User's region")
    language: str = Field("en", description="Preferred language")
    limit: int = Field(10, description="Number of recommendations", ge=1, le=50)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with centralized dependency management"""
    global embedding_manager, vector_client, similarity_search, recommendation_engine
    
    try:
        logger.info("🚀 Starting Vectorization Service...")
        
        if DEPENDENCY_MANAGER_AVAILABLE:
            # Setup SDG-specific dependencies including vectorization tasks
            await _setup_vectorization_dependencies()
            
            # Use centralized dependency management
            await dependency_manager.start_all_services()
            
            # Wait for our specific dependencies
            await wait_for_dependencies("weaviate", "database")
            
            logger.info("✅ Dependencies are ready for vectorization service")
            
        else:
            # Fallback to direct initialization
            logger.warning("Using direct initialization fallback")
            await _initialize_services_directly()
        
        # Initialize vectorization-specific services
        await _initialize_vectorization_services()
        
        logger.info("✅ Vectorization Service initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Critical error during vectorization startup: {e}")
        raise
    finally:
        logger.info("🔄 Starting Vectorization Service shutdown...")
        
        # Cleanup vectorization services
        await _cleanup_vectorization_services()
        
        # Shutdown dependency manager if available
        if DEPENDENCY_MANAGER_AVAILABLE:
            await dependency_manager.shutdown_all_services()
        
        logger.info("✅ Vectorization Service shutdown complete")

async def _setup_vectorization_dependencies():
    """Setup vectorization-specific dependencies"""
    
    async def initialize_embedding_models():
        """Initialize embedding models for vectorization service"""
        global embedding_manager
        logger.info("🤖 Initializing embedding models...")
        
        try:
            embedding_config = {
                "sentence_transformer_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "custom_model_path": os.environ.get("CUSTOM_MODEL_PATH", "bert-base-multilingual-cased")
            }
            
            embedding_manager = EmbeddingManager(embedding_config)
            logger.info("✅ Embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding models: {e}")
            raise

    async def initialize_vector_client():
        """Initialize vector database client"""
        global vector_client
        logger.info("🗄️ Initializing vector database client...")
        
        try:
            weaviate_config = {
                "url": os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080"),
                "embedded": False,
                "max_connections": int(os.environ.get("WEAVIATE_MAX_CONNECTIONS", "10")),
                "retry_attempts": int(os.environ.get("WEAVIATE_RETRY_ATTEMPTS", "3")),
                "retry_delay": float(os.environ.get("WEAVIATE_RETRY_DELAY", "1.0")),
                "api_key": os.environ.get("WEAVIATE_API_KEY"), 
            }
            
            # Validate Weaviate URL
            if not weaviate_config["url"].startswith(('http://', 'https://')):
                logger.error(f"Invalid WEAVIATE_URL format: {weaviate_config['url']}")
                weaviate_config["url"] = "http://weaviate_service:8080"
                logger.warning(f"Using fallback URL: {weaviate_config['url']}")
            
            vector_client = VectorDBClient(weaviate_config)
            logger.info("✅ Vector database client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database client: {e}")
            raise

    # Register startup tasks with dependency manager
    dependency_manager.register_startup_task("vectorization_embeddings", initialize_embedding_models)
    dependency_manager.register_startup_task("vectorization_vector_db", initialize_vector_client)
    
    # Register cleanup tasks
    async def cleanup_embedding_models():
        """Cleanup embedding models"""
        global embedding_manager
        embedding_manager = None
        logger.info("✅ Embedding models cleaned up")

    async def cleanup_vector_client():
        """Cleanup vector database client"""
        global vector_client
        if vector_client:
            try:
                vector_client.close()
            except Exception as e:
                logger.warning(f"Error closing vector client: {e}")
        vector_client = None
        logger.info("✅ Vector database client cleaned up")
    
    dependency_manager.register_cleanup_task("vectorization_embeddings", cleanup_embedding_models)
    dependency_manager.register_cleanup_task("vectorization_vector_db", cleanup_vector_client)

async def _initialize_vectorization_services():
    """Initialize vectorization-specific search services"""
    global similarity_search, recommendation_engine
    
    try:
        # Ensure we have required components
        if not embedding_manager:
            raise ValueError("Embedding manager not initialized")
        if not vector_client:
            raise ValueError("Vector client not initialized")
        
        # Initialize search services
        logger.info("🔍 Initializing similarity search...")
        similarity_search = SimilaritySearch(vector_client, embedding_manager)
        
        logger.info("🎯 Initializing recommendation engine...")
        recommendation_engine = SDGRecommendationEngine(similarity_search)
        
        logger.info("✅ Search services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search services: {e}")
        raise

async def _cleanup_vectorization_services():
    """Cleanup vectorization services"""
    global similarity_search, recommendation_engine
    
    similarity_search = None
    recommendation_engine = None
    logger.info("✅ Search services cleaned up")

async def _initialize_services_directly():
    """Fallback initialization without dependency manager"""
    global embedding_manager, vector_client
    
    logger.info("⚠️ Using direct service initialization (fallback mode)")
    
    # Load configuration
    def get_weaviate_config():
        weaviate_url = os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080")
        if not weaviate_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid WEAVIATE_URL format: {weaviate_url}")
            weaviate_url = "http://weaviate_service:8080"
            logger.warning(f"Using fallback URL: {weaviate_url}")
        
        return {
            "url": weaviate_url,
            "embedded": False,
            "max_connections": 10,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
    
    config = {
        "weaviate": get_weaviate_config(),
        "embeddings": {
            "sentence_transformer_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "openai_api_key": os.environ.get("OPENAI_API_KEY")
        }
    }
    
    # Initialize services with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            embedding_manager = EmbeddingManager(config.get("embeddings", {}))
            vector_client = VectorDBClient(config.get("weaviate", {}))
            logger.info("✅ Services initialized directly")
            break
        except Exception as e:
            logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("❌ All initialization attempts failed")
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Initialize FastAPI app
app = FastAPI(
    title="SDG Vectorization Service",
    description="Microservice for SDG content embedding generation, vector storage, and semantic search",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection functions
async def get_embedding_manager() -> EmbeddingManager:
    """Get embedding manager with dependency validation"""
    if DEPENDENCY_MANAGER_AVAILABLE:
        await wait_for_dependencies("vectorization_embeddings")
    
    if embedding_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="Embedding manager not initialized - service may still be starting"
        )
    return embedding_manager

async def get_vector_client_dep() -> VectorDBClient:
    """Get vector client with dependency validation"""
    if DEPENDENCY_MANAGER_AVAILABLE:
        await wait_for_dependencies("vectorization_vector_db", "weaviate")
    
    if vector_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Vector client not initialized - service may still be starting"
        ) 
    return vector_client

async def get_similarity_search() -> SimilaritySearch:
    """Get similarity search with dependency validation"""
    if DEPENDENCY_MANAGER_AVAILABLE:
        await wait_for_dependencies("vectorization_embeddings", "vectorization_vector_db")
    
    if similarity_search is None:
        raise HTTPException(
            status_code=503, 
            detail="Similarity search not initialized - service may still be starting"
        )
    return similarity_search

async def get_recommendation_engine() -> SDGRecommendationEngine:
    """Get recommendation engine with dependency validation"""
    if DEPENDENCY_MANAGER_AVAILABLE:
        await wait_for_dependencies("vectorization_embeddings", "vectorization_vector_db")
    
    if recommendation_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not initialized - service may still be starting"
        )
    return recommendation_engine

# Enhanced health check with dependency status
@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint with comprehensive dependency status"""
    try:
        # Check all service components
        embedding_health = embedding_manager is not None
        vector_health = vector_client.health_check() if vector_client else {"status": "unavailable"}
        search_health = similarity_search.health_check() if similarity_search else {"status": "unavailable"}
        
        # Get dependency status if available
        dependency_status = {}
        if DEPENDENCY_MANAGER_AVAILABLE:
            try:
                dependency_status = await get_dependency_status()
            except Exception as e:
                logger.warning(f"Could not get dependency status: {e}")
                dependency_status = {"status": "unknown", "error": str(e)}
        
        # Determine overall status
        component_statuses = [
            embedding_health,
            vector_health.get("status") == "healthy",
            search_health.get("status") == "healthy"
        ]
        
        overall_status = "healthy" if all(component_statuses) else "unhealthy"
        
        response = {
            "status": overall_status,
            "service": "SDG Vectorization Service",
            "version": "1.0.0",
            "timestamp": dependency_status.get("last_check"),
            "components": {
                "embedding_manager": "healthy" if embedding_health else "unhealthy",
                "vector_client": vector_health,
                "similarity_search": search_health,
                "recommendation_engine": "healthy" if recommendation_engine else "unhealthy"
            },
            "dependency_manager": {
                "available": DEPENDENCY_MANAGER_AVAILABLE,
                "overall_status": dependency_status.get("overall_status", "unknown") if dependency_status else "not_available"
            }
        }
        
        if dependency_status and "services" in dependency_status:
            response["dependencies"] = dependency_status["services"]
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error", 
                "service": "SDG Vectorization Service",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

@app.post("/embeddings", tags=["Embeddings"])
async def generate_embeddings(
    request: EmbeddingRequest,
    embedding_mgr: EmbeddingManager = Depends(get_embedding_manager)
):
    """Generate embeddings for input texts"""
    try:
        embeddings = embedding_mgr.encode(
            texts=request.texts,
            model_name=request.model_name
        )
        
        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = embeddings.tolist()
        
        return {
            "embeddings": embeddings_list,
            "dimension": len(embeddings_list[0]) if embeddings_list else 0,
            "model_used": request.model_name,
            "text_count": len(request.texts),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents", tags=["Documents"])
async def store_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    embedding_mgr: EmbeddingManager = Depends(get_embedding_manager),
    vector_db: VectorDBClient = Depends(get_vector_client_dep)
):
    """Store document with embeddings in vector database"""
    try:
        # Generate embeddings for title and content
        content_for_embedding = f"{request.title}\n\n{request.summary or request.content[:1000]}"
        embeddings = embedding_mgr.encode_sdg_content(
            content=content_for_embedding,
            sdg_goals=request.sdg_goals,
            language=request.language
        )
        
        # Prepare document for storage
        document = {
            "title": request.title,
            "content": request.content,
            "summary": request.summary,
            "sdg_goals": request.sdg_goals,
            "region": request.region,
            "language": request.language,
            "source_url": request.source_url,
            "confidence_score": 0.8,  # Default confidence
            "publication_date": "2024-01-01T00:00:00Z",  # Should come from metadata
            "vector": embeddings["embedding"]
        }
        
        # Add any additional metadata
        if request.metadata:
            document.update(request.metadata)
        
        # Store document (async in background)
        background_tasks.add_task(
            vector_db.insert_embeddings,
            documents=[document],
            class_name="SDGArticle"
        )
        
        return {
            "status": "accepted",
            "message": "Document queued for storage",
            "document_id": f"pending_{hash(request.title)}",  # Generate proper ID in production
            "embedding_dimension": embeddings["dimension"]
        }
        
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", tags=["Search"])
async def semantic_search(
    request: SearchRequest,
    search_engine: SimilaritySearch = Depends(get_similarity_search)
):
    """Perform semantic search on SDG content"""
    try:
        results = await search_engine.search(
            query=request.query,
            search_type=request.search_type,
            language=request.language,
            region=request.region,
            sdg_goals=request.sdg_goals,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "search_type": request.search_type,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    rec_engine: SDGRecommendationEngine = Depends(get_recommendation_engine)
):
    """Get SDG content recommendations based on user interests"""
    try:
        recommendations = await rec_engine.get_recommendations(
            user_interests=request.user_interests,
            region=request.region,
            language=request.language,
            limit=request.limit
        )
        
        return {
            "user_interests": request.user_interests,
            "recommendations": recommendations,
            "count": len(recommendations),
            "region": request.region,
            "language": request.language,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", tags=["Configuration"])
async def get_available_models(
    embedding_mgr: EmbeddingManager = Depends(get_embedding_manager)
):
    """Get available embedding models"""
    try:
        models_info = {}
        for model_name in embedding_mgr.models.keys():
            models_info[model_name] = embedding_mgr.get_model_info(model_name)
        
        return {
            "available_models": models_info,
            "default_model": embedding_mgr.default_model,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting model information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Statistics"])
async def get_service_stats(
    vector_db: VectorDBClient = Depends(get_vector_client_dep)
):
    """Get vectorization service statistics"""
    try:
        stats = vector_db.get_statistics()
        
        return {
            "database_statistics": stats,
            "service": "SDG Vectorization Service",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting service statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError", "service": "vectorization"}
    )

@app.exception_handler(ConnectionError)
async def connection_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable", "type": "ConnectionError", "service": "vectorization"}
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request, exc):
    return JSONResponse(
        status_code=504,
        content={"detail": "Request timeout", "type": "TimeoutError", "service": "vectorization"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
