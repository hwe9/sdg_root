from datetime import datetime
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from .database import get_db, check_database_health
from . import models, schemas
from ..core.dependency_manager import dependency_manager, setup_sdg_dependencies, get_dependency_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting SDG API Service...")
    try:
        setup_sdg_dependencies()
        # WICHTIG: Keine create_all() hier – Schema via Alembic migrieren
        await dependency_manager.start_all_services()
        logger.info("✅ SDG API Service started successfully")
        yield
    finally:
        logger.info("🔄 Shutting down SDG API Service...")
        await dependency_manager.shutdown_all_services()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SDG API Service",
    description="RESTful API for SDG data management and analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "SDG API Service is running!",
        "version": "2.0.0",
        "status": "healthy"
    }

# --- CRUD Endpoints for Articles ---

@app.post("/articles/", response_model=schemas.Article)
def create_article(article: schemas.ArticleCreate, db: Session = Depends(get_db)):
    db_article = models.Article(**article.dict())
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

@app.get("/articles/", response_model=List[schemas.Article])
def read_articles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    articles = db.query(models.Article).offset(skip).limit(limit).all()
    return articles

@app.get("/articles/{article_id}", response_model=schemas.Article)
def read_article(article_id: int, db: Session = Depends(get_db)):
    db_article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if db_article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return db_article

@app.get("/articles/{article_id}/chunks")
async def get_article_chunks(
    article_id: int,
    sdg_filter: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    query = db.query(models.ArticleChunk).filter(models.ArticleChunk.article_id == article_id)
    if sdg_filter:
        query = query.filter(models.ArticleChunk.sdg_section == sdg_filter) 
    chunks = query.order_by(models.ArticleChunk.chunk_order).limit(limit).all()
    return {
        "article_id": article_id,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

@app.get("/health")
async def health_check():
    db_healthy = check_database_health()
    dependency_status = await get_dependency_status()
    overall_status = "healthy" if (
        db_healthy and 
        dependency_status.get("overall_status") == "healthy"
    ) else "unhealthy"
    return {
        "status": overall_status,
        "service": "SDG API Service", 
        "version": "2.0.0",
        "database": "connected" if db_healthy else "disconnected",
        "dependencies": dependency_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/search/sdg/{sdg_id}/chunks")
async def search_chunks_by_sdg(
    sdg_id: int,
    query: Optional[str] = None,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    sdg = db.query(models.Sdg).filter(models.Sdg.id == sdg_id).first()
    if not sdg:
        raise HTTPException(status_code=404, detail="SDG not found")
    
    chunks_query = db.query(models.ArticleChunk).join(models.Article).filter(
        models.Article.sdg_id == sdg_id
    )
    
    if query:
        chunks_query = chunks_query.filter(models.ArticleChunk.text.contains(query))
    
    chunks = chunks_query.limit(limit).all()
    
    return {
        "sdg_id": sdg_id,
        "sdg_name": sdg.name,
        "query": query,
        "results": chunks
    }

@app.get("/articles/{article_id}/summary")
async def get_article_summary(
    article_id: int,
    max_chunks: int = 5,
    db: Session = Depends(get_db)
):
    article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Get most relevant chunks (ordered by importance/SDG relevance)
    chunks = db.query(models.ArticleChunk).filter(
        models.ArticleChunk.article_id == article_id
    ).order_by(models.ArticleChunk.chunk_order).limit(max_chunks).all()
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for article")
    
    # Generate summary from chunks
    summary_text = " ".join([chunk.text[:200] + "..." for chunk in chunks])
    
    # Extract SDG information
    sdg_sections = list(set([chunk.sdg_section for chunk in chunks if chunk.sdg_section]))
    
    return {
        "article_id": article_id,
        "article_title": article.title,
        "summary": summary_text,
        "sdg_sections": sdg_sections,
        "chunk_count": len(chunks),
        "generated_at": datetime.utcnow().isoformat()
    }

@app.post("/images/", response_model=schemas.ImageBase)
def create_image(image: schemas.ImageBase, db: Session = Depends(get_db)):
    db_image = models.Image(**image.dict())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

@app.get("/images/{article_id}", response_model=List[schemas.ImageBase])
def get_images(article_id: int, db: Session = Depends(get_db)):
    images = db.query(models.Image).filter(models.Image.article_id == article_id).all()
    return images

# --- CRUD Endpoints for SDGs ---

@app.post("/sdgs/", response_model=schemas.Sdg)
def create_sdg(sdg: schemas.SdgCreate, db: Session = Depends(get_db)):
    db_sdg = models.Sdg(**sdg.dict())
    db.add(db_sdg)
    db.commit()
    db.refresh(db_sdg)
    return db_sdg

@app.get("/sdgs/", response_model=List[schemas.Sdg])
def read_sdgs(db: Session = Depends(get_db)):
    sdgs = db.query(models.Sdg).all()
    return sdgs

# --- CRUD Endpoints for Actors ---

@app.post("/actors/", response_model=schemas.Actor)
def create_actor(actor: schemas.ActorCreate, db: Session = Depends(get_db)):
    db_actor = models.Actor(**actor.dict())
    db.add(db_actor)
    db.commit()
    db.refresh(db_actor)
    return db_actor

@app.get("/actors/", response_model=List[schemas.Actor])
def read_actors(db: Session = Depends(get_db)):
    actors = db.query(models.Actor).all()
    return actors

