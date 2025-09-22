import os
import time
from .file_handler import FileHandler
from .processing_logic import ProcessingLogic
from .db_utils import save_to_database

class ProcessingWorker:
    def __init__(self, raw_data_dir: str, processed_data_dir: str, database_url: str,
                 whisper_model=None, sentence_model=None):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.database_url = database_url
        images_root = os.environ.get("IMAGES_DIR", os.path.join(processed_data_dir, "images"))
        self.file_handler = FileHandler(images_dir=images_root)
        self.processing_logic = ProcessingLogic(whisper_model, sentence_model)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def run_worker(self):
        """Laufender Worker für alle JSON-Metadatendateien in raw_data_dir"""
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

                    # Erweiterung: Bilder aus Medien extrahieren (optional)
                    # image_data = []
                    # if media_path.endswith('.pdf'):
                    #    image_data = self.file_handler._extract_images_from_pdf(media_path, base_name)
                    # elif media_path.endswith('.docx'):
                    #    image_data = self.file_handler._extract_images_from_docx(media_path, base_name)
                    # ... an API/db_utils übergeben ...

                    # Suche nach zugehöriger Mediendatei
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

                    # Inhalt extrahieren
                    if media_path.endswith('.mp3'):
                        text_content = self.processing_logic.transcribe_audio(media_path)
                    else:
                        text_content = self.file_handler.extract_text(media_path)

                    processed_data = self.processing_logic.process_text_for_ai(text_content)
                    # Setze alle neuen Felder auf das Metadatenobjekt
                    for k, v in processed_data.items():
                        metadata[k] = v

                    # Embeddings als Output für Weaviate
                    embeddings = processed_data['embeddings']

                    # Datenbank speichern (DB + Vektor)
                    save_to_database(metadata, text_content, embeddings)

                    # Optional: Backup/Output
                    backup_path = os.path.join(self.processed_data_dir, f"{base_name}_processed.json")
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump({
                            "metadata": metadata,
                            "text": text_content,
                            "embeddings": embeddings
                        }, f, indent=4)
                    os.remove(metadata_path)
                    os.remove(media_path)
                    print(f"Verarbeitung von {json_file_name} erfolgreich. Dateien gelöscht.")

                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {json_file_name}: {e}")
                    time.sleep(5)

if __name__ == "__main__":
    RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/data/raw_data")
    PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "/data/processed_data")
    DATABASE_URL = os.environ.get("DATABASE_URL")
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer
    whisper_model = WhisperModel("small", device="cpu")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    worker = ProcessingWorker(
        raw_data_dir=RAW_DATA_DIR,
        processed_data_dir=PROCESSED_DATA_DIR,
        database_url=DATABASE_URL,
        whisper_model=whisper_model,
        sentence_model=sentence_model
    )
    worker.run_worker()
