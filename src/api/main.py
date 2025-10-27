from datetime import datetime
import logging
import asyncio
import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Depends
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from typing import Optional

from .database import get_db
from .database import check_database_health
from . import models
from . import schemas
from ..core.dependency_manager import (
    dependency_manager,
    setup_sdg_dependencies,
    get_dependency_status,
    ServiceStatus,
)
from ..core.health_utils import HealthCheckResponse
from ..core.logging_config import get_logger

logger = get_logger("api")

def _norm(name: str) -> str:
    """Canonical service key (wie im DependencyManager verwendet)."""
    return name.replace("_", "").lower()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start socket immediately; track readiness in background."""
    logger.info("🚀 Starting SDG API Service...")
    # Register deps and mark non-critical as optional
    setup_sdg_dependencies()
    for svc in ("weaviate", "weaviate_transformer",
                "vectorization", "content_extraction",
                "data_retrieval", "data_processing"):
        key = _norm(svc)
        if key in getattr(dependency_manager, "services", {}):
            dependency_manager.services[key].required = False
            logger.info(f"Marked {key} as optional for api")

    # Start dependency manager in background (non-blocking)
    asyncio.create_task(dependency_manager.start_all_services())

    # Background readiness waiter (logs only)
    async def _await_db_ready():
        db_key = _norm("database")
        start_ts = time.time()
        timeout = float(os.getenv("DB_STARTUP_MAX_WAIT_SEC", "60"))
        attempt = 0
        while time.time() - start_ts < timeout:
            attempt += 1
            status = dependency_manager.service_status.get(db_key)
            if status == ServiceStatus.HEALTHY:
                logger.info("✅ Database reported HEALTHY; API ready.")
                return
            await asyncio.sleep(min(2.0 * attempt, 5.0))
        logger.error("❌ Database not ready within bounded wait; API stays unready.")
    asyncio.create_task(_await_db_ready())
    try:
        yield
    finally:
        logger.info("🔄 Shutting down SDG API Service...")
        await dependency_manager.shutdown_all_services()
        logger.info("🛑 SDG API Service shutdown complete")

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

@app.get("/live")
def live():
    return {"status": "up", "service": "SDG API Service", "timestamp": datetime.utcnow().isoformat()}

@app.head("/live")
def live_head():
    return JSONResponse(status_code=200, content=None)

@app.get("/ready")
async def ready_check():
    try:
        db_healthy = check_database_health()
        deps = await get_dependency_status()
        services = deps.get("services", {})
        # required services für die API-Readiness (API selbst ausklammern)
        required_names = [n for n, info in services.items() if info.get("required")]
        if "api" in required_names:
            required_names.remove("api")
        required_ok = all(services.get(n, {}).get("status") == "healthy" for n in required_names)
        ready = bool(db_healthy and required_ok)
        payload = {
            "status": "ready" if ready else "not_ready",
            "service": "SDG API Service",
            "version": "2.0.0",
            "required_dependencies": {n: services.get(n, {}) for n in required_names},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return JSONResponse(status_code=200 if ready else 503, content=payload)
    except Exception as e:
        logging.exception("Readiness check error")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "service": "SDG API Service", "version": "2.0.0", "error": str(e),
                     "timestamp": datetime.utcnow().isoformat()},
        )

@app.get("/health")
async def health_check():
    """Standardized health check endpoint"""
    try:
        db_healthy = check_database_health()
        deps = await get_dependency_status()
        overall_ok = bool(db_healthy and deps.get("overall_status") == "healthy")
        
        if overall_ok:
            return JSONResponse(
                status_code=200,
                content=HealthCheckResponse.healthy_response(
                    "SDG API Service", "2.0.0",
                    components={"database": "connected"},
                    dependencies=deps
                )
            )
        else:
            return JSONResponse(
                status_code=503,
                content=HealthCheckResponse.unhealthy_response(
                    "SDG API Service", "2.0.0",
                    components={"database": "disconnected"},
                    dependencies=deps
                )
            )
    except Exception as e:
        logger.exception("Health check error")
        return JSONResponse(
            status_code=503,
            content=HealthCheckResponse.error_response("SDG API Service", "2.0.0", str(e))
        )

@app.head("/health")
async def health_head():
    try:
        db_healthy = check_database_health()
        deps = await get_dependency_status()
        overall_ok = db_healthy and deps.get("overall_status") == "healthy"
        return JSONResponse(status_code=200 if overall_ok else 503, content=None)
    except Exception:
        return JSONResponse(status_code=503, content=None)

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

# --- SDG-bezogene Suche ---

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

    # Relevante Chunks (z. B. nach chunk_order)
    chunks = db.query(models.ArticleChunk).filter(
        models.ArticleChunk.article_id == article_id
    ).order_by(models.ArticleChunk.chunk_order).limit(max_chunks).all()

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for article")

    # Simple Heuristik: anreißende Zusammenfassung
    summary_text = " ".join([chunk.text[:200] + "..." for chunk in chunks])

    # SDG-Abschnitte extrahieren
    sdg_sections = list({chunk.sdg_section for chunk in chunks if chunk.sdg_section})

    return {
        "article_id": article_id,
        "article_title": article.title,
        "summary": summary_text,
        "sdg_sections": sdg_sections,
        "chunk_count": len(chunks),
        "generated_at": datetime.utcnow().isoformat()
    }

# --- CRUD Endpoints for Images ---

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
