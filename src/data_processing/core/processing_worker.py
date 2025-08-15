# sdg_root/src/data_processing/core/processing_worker.py
import os
import time
from .file_handler import FileHandler
from .processing_logic import ProcessingLogic
from .db_utils import save_to_database

class ProcessingWorker:
    def __init__(self, raw_data_dir: str, processed_data_dir: str, database_url: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.database_url = database_url
        self.file_handler = FileHandler()
        self.processing_logic = ProcessingLogic(whisper_model, sentence_model)

    def run_worker(self):
        """Überwacht das RAW_DATA_DIR auf neue Dateien und verarbeitet sie."""
        print("Starte Data Processing Service...")
        while True:
            json_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.json')]
            if not json_files:
                print("Keine neuen Metadaten-Dateien gefunden. Warte...")
                time.sleep(30)
                continue

            for json_file_name in json_files:
                try:
                    metadata_path = os.path.join(self.raw_data_dir, json_file_name)
                    metadata = self.file_handler.get_metadata_from_json(metadata_path)
                    
                    base_name = os.path.splitext(json_file_name)[0]
                    media_path = None
                    for ext in ['.mp3', '.txt', '.pdf', '.docx', '.csv']:
                        potential_path = os.path.join(self.raw_data_dir, f"{base_name}{ext}")
                        if os.path.exists(potential_path):
                            media_path = potential_path
                            break

                    if not media_path:
                        print(f"Keine Mediendatei für {json_file_name} gefunden. Überspringe...")
                        os.remove(metadata_path)
                        continue

                    text_content = ""
                    if media_path.endswith('.mp3'):
                        text_content = self.processing_logic.transcribe_audio(media_path)
                    else:
                        text_content = self.file_handler.extract_text(media_path)
                    
                    processed_data = self.processing_logic.process_text_for_ai(text_content)
                    metadata['tags'] = processed_data['tags']
                    embeddings = processed_data['embeddings']
                    
                    save_to_database(metadata, text_content, embeddings)
                    
                    os.remove(metadata_path)
                    os.remove(media_path)
                    print(f"Verarbeitung von {json_file_name} erfolgreich. Dateien gelöscht.")

                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {json_file_name}: {e}")
                    time.sleep(10)