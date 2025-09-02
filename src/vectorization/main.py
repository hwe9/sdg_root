# src/vectorization/main.py
"""
SDG Vectorization Service - FastAPI Application
Microservice for embedding generation, vector storage, and semantic search
"""
import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

from .embedding_models import EmbeddingManager
from .vector_db_client import VectorDBClient, get_vector_client  
from .similarity_search import SimilaritySearch, SDGRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
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

# Global service instances
embedding_manager: Optional[EmbeddingManager] = None
vector_client: Optional[VectorDBClient] = None
similarity_search: Optional[SimilaritySearch] = None
recommendation_engine: Optional[SDGRecommendationEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global embedding_manager, vector_client, similarity_search, recommendation_engine
    
    try:
        # Load configuration with environment variable validation
        def get_weaviate_config():
            """Get Weaviate configuration from environment"""
            weaviate_url = os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080")
            
            # URL-Validierung
            if not weaviate_url.startswith(('http://', 'https://')):
                logger.error(f"Invalid WEAVIATE_URL format: {weaviate_url}")
                # Fallback auf Standard-URL
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
                "openai_api_key": os.environ.get("OPENAI_API_KEY")  # None ist OK
            }
        }
        
        # Log configuration (ohne sensitive Daten)
        logger.info(f"Weaviate URL: {config['weaviate']['url']}")
        logger.info(f"OpenAI API Key configured: {bool(config['embeddings']['openai_api_key'])}")
        
        # Initialize services with better error handling
        logger.info("Initializing Vectorization Service...")
        
        try:
            embedding_manager = EmbeddingManager(config.get("embeddings", {}))
            logger.info("✓ Embedding Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Embedding Manager: {e}")
            raise
        
        try:
            vector_client = VectorDBClient(config.get("weaviate", {}))
            logger.info("✓ Vector Database Client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Vector DB Client: {e}")
            raise
            
        try:
            similarity_search = SimilaritySearch(vector_client, embedding_manager)
            logger.info("✓ Similarity Search initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Similarity Search: {e}")
            raise
            
        try:
            recommendation_engine = SDGRecommendationEngine(similarity_search)
            logger.info("✓ SDG Recommendation Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Recommendation Engine: {e}")
            raise
        
        logger.info("🚀 Vectorization Service initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}")
        # Cleanup partially initialized services
        if 'vector_client' in locals() and vector_client:
            try:
                vector_client.close()
            except:
                pass
        raise
    finally:
        # Shutdown
        logger.info("🔄 Starting Vectorization Service shutdown...")
        if vector_client:
            try:
                vector_client.close()
                logger.info("✓ Vector Database Client closed")
            except Exception as e:
                logger.warning(f"Error closing vector client: {e}")
        logger.info("✅ Vectorization Service shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="SDG Vectorization Service",
    description="Microservice for SDG content embedding generation, vector storage, and semantic search",
    version="1.0.0",
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

# Dependency injection
async def get_embedding_manager() -> EmbeddingManager:
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not initialized")
    return embedding_manager

async def get_vector_client_dep() -> VectorDBClient:
    if vector_client is None:
        raise HTTPException(status_code=503, detail="Vector client not initialized") 
    return vector_client

async def get_similarity_search() -> SimilaritySearch:
    if similarity_search is None:
        raise HTTPException(status_code=503, detail="Similarity search not initialized")
    return similarity_search

async def get_recommendation_engine() -> SDGRecommendationEngine:
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    return recommendation_engine

# API Endpoints

@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint"""
    try:
        # Check all service components
        embedding_health = embedding_manager is not None
        vector_health = vector_client.health_check() if vector_client else {"status": "unavailable"}
        search_health = similarity_search.health_check() if similarity_search else {"status": "unavailable"}
        
        overall_status = "healthy" if (
            embedding_health and 
            vector_health.get("status") == "healthy" and
            search_health.get("status") == "healthy"
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "service": "SDG Vectorization Service",
            "version": "1.0.0",
            "components": {
                "embedding_manager": "healthy" if embedding_health else "unhealthy",
                "vector_client": vector_health,
                "similarity_search": search_health
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)}
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
            "text_count": len(request.texts)
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
    search_service: SimilaritySearch = Depends(get_similarity_search)
):
    """Perform semantic search across SDG content"""
    try:
        results = await search_service.semantic_search(
            query=request.query,
            search_type=request.search_type,
            language=request.language,
            region=request.region,
            sdg_goals=request.sdg_goals,
            limit=request.limit
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/sdg/{sdg_goal}", tags=["Search"])
async def search_by_sdg_goal(
    sdg_goal: int,
    limit: int = 20,
    region: Optional[str] = None,
    vector_db: VectorDBClient = Depends(get_vector_client_dep)
):
    """Search for content related to specific SDG goal"""
    try:
        if not (1 <= sdg_goal <= 17):
            raise HTTPException(status_code=400, detail="SDG goal must be between 1 and 17")
        
        results = vector_db.search_by_sdg_goals(
            sdg_goals=[sdg_goal],
            limit=limit
        )
        
        # Filter by region if specified
        if region:
            results = [r for r in results if r.get("region") == region]
        
        return {
            "sdg_goal": sdg_goal,
            "region": region,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching by SDG goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/region/{region}", tags=["Search"])
async def search_by_region(
    region: str,
    limit: int = 30,
    vector_db: VectorDBClient = Depends(get_vector_client_dep)
):
    """Search for region-specific SDG content"""
    try:
        results = vector_db.search_by_region(
            region=region,
            limit=limit
        )
        
        return {
            "region": region,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching by region: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    rec_engine: SDGRecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized SDG content recommendations"""
    try:
        # Validate SDG goals
        invalid_goals = [g for g in request.user_interests if not (1 <= g <= 17)]
        if invalid_goals:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid SDG goals: {invalid_goals}. Must be between 1 and 17."
            )
        
        recommendations = await rec_engine.recommend_content(
            user_interests=request.user_interests,
            region=request.region,
            language=request.language,
            limit=request.limit
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics", tags=["Analytics"])
async def get_statistics(
    vector_db: VectorDBClient = Depends(get_vector_client_dep)
):
    """Get vector database statistics"""
    try:
        stats = vector_db.get_statistics()
        return {
            "database_statistics": stats,
            "total_documents": sum(stats.values()),
            "available_classes": list(stats.keys())
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
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
            "default_model": embedding_mgr.default_model
        }
    except Exception as e:
        logger.error(f"Error getting model information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError"}
    )

@app.exception_handler(ConnectionError)
async def connection_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable", "type": "ConnectionError"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
