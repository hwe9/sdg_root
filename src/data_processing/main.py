import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import time
import json
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
import re
import logging
from typing import Dict, Any

from core.db_utils import save_to_database
from core.file_handler import FileHandler
from core.processing_logic import ProcessingLogic
from core.api_client import ApiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "/app/raw_data"
PROCESSED_DATA_DIR = "/app/processed_data"
DATABASE_URL = os.environ.get("DATABASE_URL")
IMAGES_DIR = "/app/images"
CLEANUP_INTERVAL_DAYS = 7 
last_cleanup_timestamp = 0

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def initialize_models_with_retry():
    """Initialize AI models with retry logic and fallbacks"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Whisper Model mit Fallback
            try:
                whisper_model = WhisperModel("small", device="cpu")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Whisper model loading failed: {e}")
                whisper_model = None  # Fallback zu None
            
            # Sentence Model (kritisch)
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence Transformer loaded successfully")
            
            return whisper_model, sentence_model
            
        except Exception as e:
            logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.critical("All model loading attempts failed")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

try:
    whisper_model, sentence_model = initialize_models_with_retry()
    logger.info("AI models initialization completed")
except Exception as e:
    logger.error(f"Critical error loading AI models: {e}")
    exit(1)

file_handler = FileHandler(IMAGES_DIR)
processing_logic = ProcessingLogic(whisper_model, sentence_model)
api_client = ApiClient()

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
    logger.info("Starting Data Processing Service...")
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
                logger.info("No new metadata files found. Waiting...")
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
                    
                    if len(text_content) > 1000:
                        processed_data = processing_logic.process_text_for_ai_with_chunking(text_content)
                        save_to_database(metadata, text_content, processed_data['combined_embeddings'], processed_data['chunks'])
                    else:
                        processed_data = processing_logic.process_text_for_ai(text_content)
                        save_to_database(metadata, text_content, processed_data['embeddings'])

                    save_backup(metadata, text_content, processed_data, base_name, PROCESSED_DATA_DIR)

                    os.remove(metadata_path)
                    os.remove(media_path)
                    logger.info(f"Processing of {json_file_name} completed successfully")

                except Exception as e:
                    logger.error(f"Error processing {json_file_name}: {e}")
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
        "processed_data": processed_data
    }
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(backup_content, f, indent=4, ensure_ascii=False)
    logger.info(f"Backup saved: {backup_path}")

if __name__ == "__main__":
    run_processing_worker()
