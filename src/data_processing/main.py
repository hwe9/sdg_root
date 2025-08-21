import os
import time
import json
import psycopg2
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from sqlalchemy.exc import OperationalError
import PyPDF2
from docx import Document
import csv
import re
import datetime

from .core.db_utils import save_to_database
from .core.file_extraction import FileHandler
from .core.processing_logic import ProcessingLogic
from .core.api_client import ApiClient


# Konfiguration
RAW_DATA_DIR = "/app/raw_data"
PROCESSED_DATA_DIR = "/app/processed_data"
DATABASE_URL = os.environ.get("DATABASE_URL")
IMAGES_DIR = "/app/images"

CLEANUP_INTERVAL_DAYS = 7 # Bereinigung alle 7 Tage
last_cleanup_timestamp = 0


# Sicherstellen, dass die Verzeichnisse existieren
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Lade KI-Modelle einmalig
try:
    whisper_model = WhisperModel("small", device="cpu")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Fehler beim Laden der KI-Modelle: {e}")
    exit(1)

file_handler = FileHandler()
processing_logic = ProcessingLogic(whisper_model, sentence_model)
api_client = ApiClient()


def run_processing_worker():
    """Überwacht das RAW_DATA_DIR auf neue Dateien und verarbeitet sie."""
    print("Starte Data Processing Service...")
    global last_cleanup_timestamp
    while True:
        # Periodische Bereinigung durchführen
        if (datetime.datetime.now() - datetime.datetime.fromtimestamp(last_cleanup_timestamp)).days >= CLEANUP_INTERVAL_DAYS:
            file_handler.cleanup_processed_data(PROCESSED_DATA_DIR, CLEANUP_INTERVAL_DAYS)
            last_cleanup_timestamp = time.time()
            
        json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
        if not json_files:
            print("Keine neuen Metadaten-Dateien gefunden. Warte...")
            time.sleep(30)
            continue

        for json_file_name in json_files:
            try:
                metadata_path = os.path.join(RAW_DATA_DIR, json_file_name)
                
                # Metadaten aus JSON laden
                metadata = file_handler.get_metadata_from_json(metadata_path)
                source_url = metadata.get('source_url', '')
                print(source_url)

                # API-basierte Metadatenextraktion (wie zuvor)
                doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)', source_url)
                isbn_match = re.search(r'ISBN(-1[03])?:?\s+((978|979)[- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,6}[- ]?\d{1,1})', source_url)
                un_match = re.search(r'(undocs\.org|un\.org)', source_url)
                oecd_match = re.search(r'(oecd-ilibrary\.org|stats\.oecd\.org)', source_url)
                wb_match = re.search(r'(worldbank\.org|data\.worldbank\.org)', source_url)
                
                api_metadata = {}
                if doi_match:
                    api_metadata = api_client.get_metadata_from_doi(doi_match.group(1))
                    api_metadata['doi'] = doi_match.group(1)
                elif isbn_match:
                    api_metadata = api_client.get_metadata_from_isbn(isbn_match.group(2))
                    api_metadata['isbn'] = isbn_match.group(2)
                elif un_match:
                    symbol = "E/2023/1"
                    api_metadata = api_client.get_metadata_from_un_digital_library(symbol)
                elif oecd_match:
                    dataset_id = "EDU_ENRL"
                    api_metadata = api_client.get_metadata_from_oecd(dataset_id)
                elif wb_match:
                    indicator_id = "SP.POP.TOTL"
                    api_metadata = api_client.get_metadata_from_world_bank(indicator_id)
                
                metadata.update(api_metadata)
                
                # Bestimme den Pfad zur Mediendatei
                base_name = os.path.splitext(json_file_name)[0]
                media_path = None
                for ext in ['.mp3', '.txt', '.pdf', '.docx', '.csv']:
                    potential_path = os.path.join(RAW_DATA_DIR, f"{base_name}{ext}")
                    if os.path.exists(potential_path):
                        media_path = potential_path
                        break

                if not media_path:
                    print(f"Keine Mediendatei für {json_file_name} gefunden. Überspringe...")
                    os.remove(metadata_path)
                    continue

                # Text extrahieren / Audio transkribieren
                text_content = ""
                if media_path.endswith('.mp3'):
                    text_content = processing_logic.transcribe_audio(media_path)
                else:
                    text_content = file_handler.extract_text(media_path)
                
                processed_data = processing_logic.process_text_for_ai(text_content)
                metadata['tags'] = processed_data['tags']
                metadata['abstract'] = processed_data['abstract']
                embeddings = processed_data['embeddings']

                # Speichern in DB
                save_to_database(metadata, text_content, embeddings)

                # **Speichern als JSON-Backup in PROCESSED_DATA_DIR**
                backup_path = os.path.join(PROCESSED_DATA_DIR, f"{base_name}_processed.json")
                backup_content = {
                    "metadata": metadata,
                    "text": text_content,
                    "embeddings": embeddings
                }
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_content, f, indent=4)
                print(f"Backup gespeichert: {backup_path}")

                # Rohdateien löschen
                os.remove(metadata_path)
                os.remove(media_path)
                print(f"Verarbeitung von {json_file_name} erfolgreich. Rohdateien gelöscht.")

            except Exception as e:
                print(f"Fehler beim Verarbeiten von {json_file_name}: {e}")
                time.sleep(10)
      
if __name__ == "__main__":
    run_processing_worker()