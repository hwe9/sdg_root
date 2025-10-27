import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import time
import datetime
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
import re
import logging
import asyncio
from typing import List
from typing import Dict
from typing import Any
from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi import HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core.db_utils import save_to_database
from core.file_handler import FileHandler
from core.processing_logic import ProcessingLogic
from core.api_client import ApiClient
from ..core.health_utils import HealthCheckResponse

# Add after existing imports - DEPENDENCY MANAGEMENT INTEGRATION
from ..core.dependency_manager import dependency_manager
from ..core.dependency_manager import wait_for_dependencies
from ..core.dependency_manager import setup_sdg_dependencies
from ..core.logging_config import get_logger

logger = get_logger("data_processing")

from ..core.db_utils import get_database_url

RAW_DATA_DIR = "/data/raw_data"
PROCESSED_DATA_DIR = "/data/processed_data"
DATABASE_URL = get_database_url()
IMAGES_DIR = "/data/images"
CLEANUP_INTERVAL_DAYS = 7 
last_cleanup_timestamp = 0

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Global services
processing_logic = None
whisper_model = None
sentence_model = None
file_handler = None
api_client = None

# Pydantic models for API endpoints
class ProcessContentRequest(BaseModel):
    content_items: List[Dict[str, Any]]

class ProcessingResponse(BaseModel):
    success: bool
    processed_count: int
    message: str

# Initialize models with retry logic
def initialize_models_with_retry():
    """Initialize AI models with retry logic and fallbacks"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Whisper Model with fallback
            try:
                whisper_model = WhisperModel("small", device="cpu")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Whisper model loading failed: {e}")
                whisper_model = None  # Fallback to None
            
            # Sentence Model (critical)
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence Transformer loaded successfully")
            
            return whisper_model, sentence_model
            
        except Exception as e:
            logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.critical("All model loading attempts failed")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# SDG Semantic Chunker for large documents
class SDGSemanticChunker:
    def __init__(self, target_chunk_size=400, overlap_size=40):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
    
    def smart_chunk(self, text: str, preserve_sdg_context=True) -> List[Dict[str, Any]]:
        """Intelligently chunk text while preserving SDG context"""
        sentences = re.split(r'[.!?]', text)
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.target_chunk_size and current_chunk:
                # Create chunk with metadata
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "word_count": current_size,
                    "semantic_coherence": self._calculate_coherence(current_chunk),
                    "sdg_density": self._calculate_sdg_density(current_chunk) if preserve_sdg_context else 0.5
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk.split()[-self.overlap_size:]
                current_chunk = " ".join(overlap_text) + " " + sentence
                current_size = len(overlap_text) + sentence_size
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "word_count": current_size,
                "semantic_coherence": self._calculate_coherence(current_chunk),
                "sdg_density": self._calculate_sdg_density(current_chunk) if preserve_sdg_context else 0.5
            })
        
        return chunks
    
    def _calculate_coherence(self, text: str) -> float:
        """Simple coherence score based on sentence connectivity"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple heuristic: check for connecting words
        connecting_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
        connections = sum(1 for sentence in sentences for word in connecting_words if word in sentence.lower())
        
        return min(0.3 + (connections / len(sentences)), 1.0)
    
    def _calculate_sdg_density(self, text: str) -> float:
        """Calculate SDG-related keyword density"""
        sdg_keywords = ['sustainable', 'development', 'goal', 'target', 'poverty', 'hunger', 'health', 
                       'education', 'gender', 'water', 'energy', 'growth', 'infrastructure', 
                       'inequality', 'climate', 'ocean', 'biodiversity', 'peace', 'partnership']
        
        words = text.lower().split()
        sdg_count = sum(1 for word in words if any(keyword in word for keyword in sdg_keywords))
        
        return min(sdg_count / len(words), 1.0) if words else 0.0

# Initialize services
def initialize_services():
    """Initialize all services with proper error handling"""
    global processing_logic, whisper_model, sentence_model, file_handler, api_client
    
    try:
        whisper_model, sentence_model = initialize_models_with_retry()
        processing_logic = ProcessingLogic(whisper_model, sentence_model)
        file_handler = FileHandler(IMAGES_DIR)
        api_client = ApiClient()
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        return False

# Enhanced document processing function
def process_large_document(text_content: str, metadata: dict) -> dict:
    """Process large documents with intelligent chunking"""
    
    word_count = len(text_content.split())
    
    if word_count <= 200:
        # Small document - process normally
        return processing_logic.process_text_for_ai(text_content)
    
    # Large document - use smart chunking
    chunker = SDGSemanticChunker(target_chunk_size=400, overlap_size=40)
    chunks = chunker.smart_chunk(text_content, preserve_sdg_context=True)
    
    # Process each chunk
    processed_chunks = []
    all_embeddings = []
    
    for chunk in chunks:
        chunk_processed = processing_logic.process_text_for_ai(chunk["text"])
        chunk.update(chunk_processed)
        processed_chunks.append(chunk)
        all_embeddings.append(chunk_processed["embeddings"])
    
    # Combine embeddings (weighted average by chunk quality)
    weights = [chunk.get("semantic_coherence", 0.5) for chunk in processed_chunks]
    if all_embeddings:
        combined_embedding = np.average(all_embeddings, axis=0, weights=weights).tolist()
    else:
        combined_embedding = []
    
    return {
        "chunks": processed_chunks,
        "combined_embeddings": combined_embedding,
        "total_chunks": len(processed_chunks),
        "total_words": word_count,
        "processing_method": "smart_semantic_chunking"
    }

# FastAPI lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting SDG Data Processing Service...")
    
    if not initialize_services():
        logger.error("❌ Failed to initialize services")
        raise RuntimeError("Service initialization failed")
    
    logger.info("✅ SDG Data Processing Service started successfully")
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down SDG Data Processing Service...")
    logger.info("✅ SDG Data Processing Service shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SDG Data Processing Service",
    description="Microservice for processing SDG-related content with AI models",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced health check with dependency status
@app.get("/health")
async def health_check():
    """Standardized health check endpoint"""
    try:
        from ..core.dependency_manager import get_dependency_status
        
        dependency_status = await get_dependency_status()
        db_healthy = check_database_health()
        models_ok = all([processing_logic, whisper_model, sentence_model, file_handler, api_client])
        
        overall_status = "healthy" if (db_healthy and models_ok) else "unhealthy"
        
        if overall_status == "healthy":
            return JSONResponse(
                status_code=200,
                content=HealthCheckResponse.healthy_response(
                    "SDG Data Processing Service", "2.0.0",
                    components={
                        "database": "connected",
                        "models": "loaded",
                        "processing_logic": "ready",
                        "file_handler": "ready",
                        "api_client": "ready"
                    },
                    dependencies=dependency_status
                )
            )
        else:
            return JSONResponse(
                status_code=503,
                content=HealthCheckResponse.unhealthy_response(
                    "SDG Data Processing Service", "2.0.0",
                    components={
                        "database": "connected" if db_healthy else "disconnected",
                        "models": "loaded" if models_ok else "not_loaded"
                    },
                    dependencies=dependency_status
                )
            )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content=HealthCheckResponse.error_response("SDG Data Processing Service", "2.0.0", str(e))
        )

@app.post("/process-content", response_model=ProcessingResponse)
async def process_content_endpoint(request: ProcessContentRequest):
    """Process content items through the pipeline"""
    try:
        processed_count = 0
        
        for item in request.content_items:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            
            if len(content) > 1000:
                processed_data = process_large_document(content, metadata)
                save_to_database(metadata, content, processed_data['combined_embeddings'], processed_data.get('chunks'))
            else:
                processed_data = processing_logic.process_text_for_ai(content)
                save_to_database(metadata, content, processed_data['embeddings'])
            
            processed_count += 1
        
        return ProcessingResponse(
            success=True,
            processed_count=processed_count,
            message=f"Successfully processed {processed_count} content items"
        )
        
    except Exception as e:
        logger.error(f"Error processing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def check_database_health():
    """Simple database health check"""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def run_processing_worker():
    """Monitor RAW_DATA_DIR for new files and process them."""
    logger.info("Starting Data Processing Worker...")
    global last_cleanup_timestamp
    
    while True:
        try:
            current_time = time.time()
            if (current_time - last_cleanup_timestamp) > (CLEANUP_INTERVAL_DAYS * 24 * 3600):
                file_handler.cleanup_processed_data(PROCESSED_DATA_DIR, CLEANUP_INTERVAL_DAYS)
                last_cleanup_timestamp = current_time

            # Simple health check
            if not check_database_health():
                logger.warning("Database not available, waiting...")
                time.sleep(30)
                continue
                
            json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
            if not json_files:
                logger.debug("No new metadata files found. Waiting...")
                time.sleep(30)
                continue

            for json_file_name in json_files:
                try:
                    metadata_path = os.path.join(RAW_DATA_DIR, json_file_name)
                    
                    metadata = file_handler.get_metadata_from_json(metadata_path)
                    source_url = metadata.get('source_url', '')
                    logger.info(f"Processing: {source_url}")

                    api_metadata = extract_api_metadata(source_url, api_client)
                    metadata.update(api_metadata)
                    
                    base_name = os.path.splitext(json_file_name)[0]
                    media_path = find_media_file(base_name, RAW_DATA_DIR)

                    if not media_path:
                        logger.warning(f"No media file found for {json_file_name}. Skipping...")
                        os.remove(metadata_path)
                        continue

                    text_content = extract_content(media_path, file_handler, processing_logic)
                    
                    if not text_content:
                        logger.warning(f"No content extracted from {media_path}")
                        continue
                    
                    # Use enhanced processing
                    if len(text_content) > 1000:
                        processed_data = process_large_document(text_content, metadata)
                        save_to_database(metadata, text_content, processed_data['combined_embeddings'], processed_data.get('chunks'))
                    else:
                        processed_data = processing_logic.process_text_for_ai(text_content)
                        save_to_database(metadata, text_content, processed_data['embeddings'])

                    save_backup(metadata, text_content, processed_data, base_name, PROCESSED_DATA_DIR)

                    os.remove(metadata_path)
                    os.remove(media_path)
                    logger.info(f"✅ Processing of {json_file_name} completed successfully")

                except Exception as e:
                    logger.error(f"❌ Error processing {json_file_name}: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Processing worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in processing worker: {e}")
            time.sleep(60)

def extract_api_metadata(source_url: str, api_client: ApiClient) -> dict:
    """Extract metadata from various APIs based on URL patterns."""
    doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)', source_url)
    isbn_match = re.search(r'ISBN(-1[03])?:?\s+((978|979)[- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,6}[- ]?\d{1,1})', source_url)
    
    if doi_match:
        metadata = api_client.get_metadata_from_doi(doi_match.group(1))
        metadata['doi'] = doi_match.group(1)
        return metadata
    elif isbn_match:
        metadata = api_client.get_metadata_from_isbn(isbn_match.group(2))
        metadata['isbn'] = isbn_match.group(2)
        return metadata
    elif re.search(r'(undocs\.org|un\.org)', source_url):
        return api_client.get_metadata_from_un_digital_library("E/2023/1")
    elif re.search(r'(oecd-ilibrary\.org|stats\.oecd\.org)', source_url):
        return api_client.get_metadata_from_oecd("EDU_ENRL")
    elif re.search(r'(worldbank\.org|data\.worldbank\.org)', source_url):
        return api_client.get_metadata_from_world_bank("SP.POP.TOTL")
    
    return {}

def find_media_file(base_name: str, data_dir: str) -> str:
    """Find media file with given base name."""
    for ext in ['.mp3', '.txt', '.pdf', '.docx', '.csv']:
        potential_path = os.path.join(data_dir, f"{base_name}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    return None

def extract_content(media_path: str, file_handler: FileHandler, processing_logic: ProcessingLogic) -> str:
    """Extract content from media file."""
    if media_path.endswith('.mp3'):
        return processing_logic.transcribe_audio(media_path)
    else:
        return file_handler.extract_text(media_path)

def save_backup(metadata: dict, text_content: str, processed_data: dict, base_name: str, processed_dir: str):
    """Save processed data as JSON backup."""
    backup_path = os.path.join(processed_dir, f"{base_name}_processed.json")
    backup_content = {
        "metadata": metadata,
        "text": text_content,
        "processed_data": processed_data,
        "backup_timestamp": datetime.datetime.utcnow().isoformat()
    }
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(backup_content, f, indent=4, ensure_ascii=False)
    logger.info(f"Backup saved: {backup_path}")

# Modified main function with dependency management
if __name__ == "__main__":
    import threading
    import uvicorn
    
    async def start_with_dependencies():
        """Start service with dependency management"""
        logger.info("🚀 Starting Data Processing Service...")
        
        # Setup dependencies
        setup_sdg_dependencies()
        
        # Register processing-specific startup tasks
        async def initialize_processing_dependencies():
            """Initialize data processing dependencies"""
            await wait_for_dependencies("database", "weaviate", "data_retrieval")
            
            # Initialize models with dependency validation
            global processing_logic
            if not processing_logic:
                whisper_model, sentence_model = initialize_models_with_retry()
                processing_logic = ProcessingLogic(whisper_model, sentence_model)
            
            logger.info("✅ Data processing dependencies initialized")
        
        dependency_manager.register_startup_task("data_processing", initialize_processing_dependencies)
        
        # Start dependency manager
        await dependency_manager.start_all_services()
        
        # Start the worker in background
        worker_thread = threading.Thread(target=run_processing_worker, daemon=True)
        worker_thread.start()
        
        # Start FastAPI
        uvicorn.run("main:app", host="0.0.0.0", port=8001)
    
    # Run with dependency management
    asyncio.run(start_with_dependencies())
