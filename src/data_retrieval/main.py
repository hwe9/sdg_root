import os
import requests
import json
import time
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

# Definierte Pfade und Dateinamen
DATA_DIR = "/app/raw_data"
SOURCES_FILE = "quelle.txt"
os.makedirs(DATA_DIR, exist_ok=True)

# Standard yt-dlp Optionen für bessere Authentifizierung
YDL_OPTS_BASE = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'add-header': [
        'Referer:https://www.youtube.com'
    ],
    'no-check-certificate': True,
    'ignoreerrors': True,
    'sleep_requests_min': 5,
    'sleep_requests_max': 15,
}

def load_static_sources(file_path: str) -> set:
    """Läd eine Liste von URLs aus einer Textdatei."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                # Überprüfe, ob es sich um eine Kanal- oder Playlist-URL handelt
                if "youtube.com/channel" in url or "youtube.com/playlist" in url or "youtube.com/@" in url:
                    print(f"Extrahiere Video-URLs aus Kanal oder Playlist: {url}")
                    ydl_opts_info = {
                        **YDL_OPTS_BASE,
                        'skip_download': True,
                        'quiet': True,
                        'extract_flat': True
                    }
                    try:
                        with YoutubeDL(ydl_opts_info) as ydl:
                            info = ydl.extract_info(url, download=False)
                            if 'entries' in info:
                                for entry in info['entries']:
                                    if entry:
                                        all_urls.add(entry['url'])
                    except (DownloadError, ExtractorError) as e:
                        print(f"Fehler beim Extrahieren der Video-URLs von {url}: {e}")
                else:
                    all_urls.add(url)
    return all_urls

def get_metadata_and_filename(url: str, metadata: dict) -> tuple:
    """Erstellt einen eindeutigen und selbsterklärenden Dateinamen."""
    sdg_id = "SDG_placeholder"
    date_str = time.strftime("%Y%m%d")
    title_slug = metadata.get('title', 'untitled').replace(' ', '_').lower()[:30]
    return f"{sdg_id}_{date_str}_{title_slug}", title_slug

def is_sponsored(info: dict) -> bool:
    """Überprüft, ob ein Video gesponserte Schlüsselwörter enthält."""
    sponsored_keywords = ["sponsored", "werbung", "anzeige", "ad", "paid partnership"]
    title = info.get('title', '').lower()
    description = info.get('description', '').lower()
    
    for keyword in sponsored_keywords:
        if keyword in title or keyword in description:
            return True
    return False

# sdg_root/src/data_retrieval/main.py
# (ersetze nur die download_youtube_content-Funktion mit diesem Code)

def download_youtube_content(url: str):
    """
    Lädt nur YouTube-Transkripte als .vtt-Datei herunter.
    Videos ohne Transkript werden übersprungen.
    """
    ydl_opts_meta = {**YDL_OPTS_BASE, 'skip_download': True}
    filename_base = None
    
    try:
        with YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(url, download=False)

        if is_sponsored(info):
            print(f"Überspringe gesponserten Inhalt: {info.get('title')}")
            return

        filename_base, _ = get_metadata_and_filename(url, info)
        
        # Check if file exists to prevent re-download
        if os.path.exists(os.path.join(DATA_DIR, f"{filename_base}.json")):
            print(f"Datei für {url} existiert bereits. Überspringe.")
            return

        # Versuche Transkripte als .vtt herunterzuladen
        if info.get('automatic_captions') or info.get('subtitles'):
            print(f"Transkript für {url} gefunden. Starte Download...")
            metadata_file_path = os.path.join(DATA_DIR, f"{filename_base}.json")
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            print(f"Metadaten gespeichert: {metadata_file_path}")

            ydl_opts_subs = {
                **YDL_OPTS_BASE,
                'writesubtitles': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'outtmpl': os.path.join(DATA_DIR, f"{filename_base}.%(ext)s")
            }
            with YoutubeDL(ydl_opts_subs) as ydl_subs:
                ydl_subs.download([url])
            print(f"Transkript für {url} erfolgreich heruntergeladen.")
        else:
            print(f"Kein Transkript für {url} gefunden. Überspringe Download.")
            return

    except (DownloadError, ExtractorError) as e:
        print(f"Download-Fehler (Metadaten oder Transkript): {e}. Überspringe.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}. Überspringe.")
        return

def download_generic_content(url: str):
    """Lädt allgemeine Inhalte (PDF, TXT) herunter."""
    print(f"Lade Inhalt von {url} herunter...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filename_base, _ = get_metadata_and_filename(url, {'title': os.path.basename(url)})
        file_extension = url.split('.')[-1]
        file_path = os.path.join(DATA_DIR, f"{filename_base}.{file_extension}")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Inhalt gespeichert: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Download von {url}: {e}")

def run_retrieval_worker():
    """Haupt-Worker-Funktion, die den Download-Prozess steuert."""
    print("Starte Data Retrieval Service...")
    static_sources = load_static_sources(os.path.join("/app", SOURCES_FILE))
    all_sources = static_sources

    if not all_sources:
        print("Keine Quellen zum Herunterladen gefunden. Warte...")
    else:
        for url in all_sources:
            if "youtube.com" in url:
                download_youtube_content(url)
            else:
                download_generic_content(url)
    print("Data Retrieval Service hat seine Arbeit beendet.")

if __name__ == "__main__":
    while True:
        try:
            run_retrieval_worker()
            print("Warte 60 Minuten bis zum nächsten Durchlauf...")
            time.sleep(3600)
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            time.sleep(60)