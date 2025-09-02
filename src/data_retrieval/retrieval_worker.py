# src/data_retrieval/retrieval_worker.py

import os
import json
import requests
import csv
import fcntl
from datetime import datetime
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError
from urllib.parse import quote
from ..core.url_validator import url_validator
import logging
logging.basicConfig(level=logging.INFO)

class RetrievalWorker:
    def __init__(self, sources_file, data_dir, processed_file):
        self.sources_file = sources_file
        self.data_dir = data_dir
        self.processed_file = processed_file
        self.downloaded_urls_file = os.path.join(data_dir, "downloaded_urls.csv")

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SDG-Pipeline-Bot/1.0 (+https://sdg-pipeline.org/bot)'
        })
        
        # Security settings
        self.session.max_redirects = 3
        self.timeout = 15
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit

        print("---retrievalWorker instanz gestartet")
    
    def load_downloaded_urls(self):
        """Lade bereits heruntergeladene URLs aus CSV."""
        downloaded_urls = set()
        if os.path.exists(self.downloaded_urls_file):
            try:
                with open(self.downloaded_urls_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        downloaded_urls.add(row['url'])
                logging.info(f"✅ {len(downloaded_urls)} bereits heruntergeladene URLs geladen")
            except Exception as e:
                logging.error(f"❌ Fehler beim Laden der URL-Historie: {e}")
        return downloaded_urls

    def save_downloaded_url(self, url, filename, status="success"):
        """Speichere erfolgreich heruntergeladene URL in CSV."""
        try:
            file_exists = os.path.exists(self.downloaded_urls_file)
            with open(self.downloaded_urls_file, 'a', encoding='utf-8', newline='') as f:
                fieldnames = ['url', 'filename', 'timestamp', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'url': url,
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'status': status
                })
            logging.info(f"✅ URL gespeichert in Historie: {url}")
        except Exception as e:
            logging.error(f"❌ Fehler beim Speichern der URL: {e}")

    def run(self):
        all_urls = self.load_sources()
        downloaded_urls = self.load_downloaded_urls()
        processed_data = []
        new_urls = all_urls - downloaded_urls

        if not new_urls:
            logging.info("🔄 Alle URLs bereits heruntergeladen. Keine neuen URLs zu verarbeiten.")
            return
        
        logging.info(f"📥 {len(new_urls)} neue URLs zu verarbeiten von {len(all_urls)} gesamt")

        for url in new_urls:
            try:
                if self.is_youtube(url):
                    item = self.download_youtube_content(url)
                else:
                    item = self.download_generic_content(url)
                if item:
                    processed_data.append(item)
                    self.save_downloaded_url(url, item['title'])
                    meta_filename = os.path.splitext(item['title'])[0].replace(" ", "_").replace("%20", "_") + ".json"
                    meta_path = os.path.join(self.data_dir, meta_filename)
                    with open(meta_path, "w", encoding="utf-8") as meta_f:
                        json.dump(item, meta_f, indent=2, ensure_ascii=False)
                    logging.info(f"✅ Metadaten gespeichert: {meta_path}")
                else:
                    self.save_downloaded_url(url, "failed", "failed")
            except Exception as e:
                self.handle_errors(url, e)
                self.save_downloaded_url(url, "error", "error")
        self.save_to_file(processed_data)
        self.signal_processing(processed_data)

    def load_sources(self):
        all_urls = set()
        if os.path.exists(self.sources_file):
            with open(self.sources_file, "r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        all_urls.add(url)
        logging.info(f"📋 {len(all_urls)} URLs aus Quellenliste geladen")
        return all_urls

    def is_youtube(self, url):
        return "youtube.com" in url or "youtu.be" in url

    def download_youtube_content(self, url):
        # Analog zu deinem main.py mit yt-dlp
        ydl_opts = {"skip_download": True, "quiet": True}
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return {
                "url": url,
                "title": info.get("title"),
                "description": info.get("description"),
                "metadata": info,
                "source_url": url
            }
        except (DownloadError, ExtractorError) as e:
            print(f"Error: {e}")
            return None

    def download_generic_content(self, url: str):
        """Download content with comprehensive security validation"""
        logging.info(f"Validating URL: {url}")
        
        # Validate URL first
        is_valid, error_msg = url_validator.validate_url(url)
        if not is_valid:
            logging.error(f"URL validation failed for {url}: {error_msg}")
            return None
        
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Make request with security headers
            response = self.session.get(
                url, 
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,  # Stream for size checking
                headers={
                    'Accept': 'application/pdf,text/html,application/xml,text/plain',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'close'
                }
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            allowed_types = ['application/pdf', 'text/html', 'text/plain', 'application/xml']
            if not any(allowed_type in content_type for allowed_type in allowed_types):
                logging.error(f"Disallowed content type: {content_type}")
                return None
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                logging.error(f"File too large: {content_length} bytes")
                return None
            
            # Download with size limit
            filename = self._generate_safe_filename(url)
            file_path = os.path.join(self.data_dir, filename)
            
            downloaded_size = 0
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_file_size:
                            f.close()
                            os.remove(file_path)
                            logging.error(f"File size exceeded limit during download")
                            return None
                        f.write(chunk)
            
            file_size = os.path.getsize(file_path)
            logging.info(f"✅ Download successful: {filename} ({file_size} bytes)")
            
            return {
                "url": url,
                "title": filename,
                "file_path": file_path,
                "source_url": url,
                "content_type": content_type,
                "file_size": file_size
            }
            
        except requests.exceptions.Timeout:
            logging.error(f"Timeout downloading {url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error downloading {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error downloading {url}: {e}")
            return None
    
    def _generate_safe_filename(self, url: str) -> str:
        """Generate safe filename from URL"""
        import hashlib
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        path_part = parsed.path.split('/')[-1] if parsed.path else 'download'
        
        # Sanitize filename
        safe_chars = '-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        safe_filename = ''.join(c for c in path_part if c in safe_chars)
        
        if not safe_filename or len(safe_filename) < 3:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            safe_filename = f"download_{url_hash}"
        
        # Add extension if missing
        if '.' not in safe_filename:
            safe_filename += '.pdf'  # Default extension

        return safe_filename
        

    def save_to_file(self, data):
        with open(self.processed_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def signal_processing(self, processed_data):
        # Hier reicht in der Baseline das Ablegen der JSON (File-Drop)
        print(f"Signal für Processing-Service: {self.processed_file}")

    def handle_errors(self, url, error):
        print(f"Fehler bei {url}: {error}")
