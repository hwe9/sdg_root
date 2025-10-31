import os
import hashlib
import logging
from typing import Dict
from typing import Any
from typing import Optional
from typing import List
from datetime import datetime
from urllib.parse import urlparse
import aiofiles
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadStrategy:
    
    def __init__(self):
        self.session = None
        self.timeout = int(os.getenv("HTTP_TOTAL_TIMEOUT", "45"))
        self.connect_timeout = int(os.getenv("HTTP_CONNECT_TIMEOUT", "10"))
        self.read_timeout = int(os.getenv("HTTP_READ_TIMEOUT", "30"))
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024)))
        self.user_agent = os.getenv("RETRIEVAL_UA", 'SDG-Pipeline-Bot/2.0 (+https://sdg-pipeline.org/bot)')
    
    async def initialize(self):
        connector = aiohttp.TCPConnector(
            limit=int(os.getenv("HTTP_CONN_LIMIT", "50")),
            limit_per_host=int(os.getenv("HTTP_CONN_PER_HOST", "5"))
        )
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=self.connect_timeout,
            sock_read=self.read_timeout,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.user_agent}
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    def generate_safe_filename(self, url: str, content_type: str = "") -> str:
        """Generate safe filename from URL and content type"""
        parsed = urlparse(url)
        path_part = parsed.path.split('/')[-1] if parsed.path else 'download'
        
        # Sanitize filename
        safe_chars = '-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        safe_filename = ''.join(c for c in path_part if c in safe_chars)
        
        if not safe_filename or len(safe_filename) < 3:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            safe_filename = f"download_{url_hash}"
        
        # Add appropriate extension based on content type
        if not '.' in safe_filename:
            if 'pdf' in content_type:
                safe_filename += '.pdf'
            elif 'html' in content_type:
                safe_filename += '.html'
            elif 'xml' in content_type:
                safe_filename += '.xml'
            else:
                safe_filename += '.txt'
        
        return safe_filename

class HTTPDownloadStrategy(DownloadStrategy):
    """HTTP/HTTPS download with security validation"""
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Download HTTP/HTTPS content securely"""
        try:
            # Make request with security headers
            async with self.session.get(
                url,
                allow_redirects=True,
                headers={
                    'Accept': 'application/pdf,text/html,application/xml,text/plain,application/json',  # <- JSON ergänzt
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'close'
                }
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get('content-type', '').lower()
                allowed_types = ['application/pdf', 'text/html', 'text/plain', 'application/xml', 'application/json']  # <- JSON ergänzt

                if not any(allowed_type in content_type for allowed_type in allowed_types):
                    logger.warning(f"Disallowed content type: {content_type}")
                    return None
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    logger.warning(f"File too large: {content_length} bytes")
                    return None
                
                # Generate filename
                filename = self.generate_safe_filename(url, content_type)
                file_path = os.path.join(data_dir, filename)
                
                # Download with size checking
                downloaded_size = 0
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_file_size:
                            await f.close()
                            os.remove(file_path)
                            logger.warning(f"File size exceeded during download")
                            return None
                        await f.write(chunk)
                
                file_size = os.path.getsize(file_path)
                
                return {
                    "title": filename,
                    "file_path": file_path,
                    "content_type": content_type,
                    "file_size": file_size,
                    "download_method": "http",
                    "downloaded_at": datetime.utcnow().isoformat()
                }
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error downloading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return None

class YouTubeDownloadStrategy(DownloadStrategy):
    """YouTube content extraction with filtering for Shorts and sponsored content"""
    
    def _is_shorts_video(self, url: str, info: Dict[str, Any]) -> bool:
        """Check if video is a YouTube Short"""
        # Check URL for /shorts/ path
        if "/shorts/" in url.lower():
            return True
        
        # Check duration (Shorts are typically under 60 seconds)
        duration = info.get("duration")
        if duration and duration < 60:
            return True
        
        # Check category
        categories = info.get("categories", [])
        if categories and any("short" in str(cat).lower() for cat in categories):
            return True
        
        # Check tags
        tags = info.get("tags", [])
        if tags and any("short" in str(tag).lower() for tag in tags):
            return True
        
        return False
    
    def _is_sponsored_video(self, info: Dict[str, Any]) -> bool:
        """Check if video is sponsored content"""
        # Check tags for sponsor-related keywords
        tags = info.get("tags", [])
        sponsor_keywords = ["sponsor", "sponsored", "advertisement", "ad", "promo", "promotional", "paid", "partnership"]
        if tags:
            tag_text = " ".join(str(tag).lower() for tag in tags)
            if any(keyword in tag_text for keyword in sponsor_keywords):
                return True
        
        # Check description for sponsor mentions
        description = info.get("description", "").lower()
        sponsor_patterns = [
            "sponsored by",
            "sponsor:",
            "this video is sponsored",
            "paid partnership",
            "advertisement",
            "thanks to [",
            "in partnership with"
        ]
        if any(pattern in description for pattern in sponsor_patterns):
            return True
        
        # Check for SponsorBlock chapters (if available)
        sponsorblock_chapters = info.get("sponsorblock_chapters")
        if sponsorblock_chapters:
            # If SponsorBlock data exists and has sponsor segments, it's likely sponsored
            return True
        
        return False
    
    def _extract_transcript(self, info: Dict[str, Any], ydl: Any) -> Optional[str]:
        """Extract transcript/subtitles from YouTube video if available"""
        try:
            # Try to get auto-generated or manual subtitles
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            # Preferred languages in order (English first, then others)
            preferred_langs = ['en', 'en-US', 'en-GB', 'de', 'fr', 'es', 'hi', 'zh', 'zh-CN', 'zh-TW', 'hi-IN']
            
            # Combine subtitles and automatic captions
            all_captions = {}
            if subtitles:
                all_captions.update(subtitles)
            if automatic_captions:
                all_captions.update(automatic_captions)
            
            if not all_captions:
                return None
            
            # Try preferred languages first
            for lang in preferred_langs:
                if lang in all_captions:
                    caption_list = all_captions[lang]
                    if caption_list:
                        # Get the best format (usually the first one)
                        caption_url = caption_list[0].get('url')
                        if caption_url:
                            try:
                                # Download subtitle content
                                import urllib.request
                                with urllib.request.urlopen(caption_url, timeout=10) as response:
                                    subtitle_data = response.read().decode('utf-8')
                                
                                # Parse WebVTT or SRT format and extract text
                                transcript_text = self._parse_subtitle_format(subtitle_data)
                                if transcript_text:
                                    logger.info(f"✅ Extracted transcript in {lang}")
                                    return transcript_text
                            except Exception as e:
                                logger.debug(f"Failed to download caption for {lang}: {e}")
                                continue
            
            # If no preferred language found, try any available language
            for lang, caption_list in all_captions.items():
                if caption_list:
                    caption_url = caption_list[0].get('url')
                    if caption_url:
                        try:
                            import urllib.request
                            with urllib.request.urlopen(caption_url, timeout=10) as response:
                                subtitle_data = response.read().decode('utf-8')
                            transcript_text = self._parse_subtitle_format(subtitle_data)
                            if transcript_text:
                                logger.info(f"✅ Extracted transcript in {lang}")
                                return transcript_text
                        except Exception as e:
                            logger.debug(f"Failed to download caption for {lang}: {e}")
                            continue
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting transcript: {e}")
            return None
    
    def _parse_subtitle_format(self, subtitle_data: str) -> str:
        """Parse WebVTT or SRT subtitle format and extract text"""
        lines = subtitle_data.split('\n')
        text_parts = []
        
        for line in lines:
            line = line.strip()
            # Skip timing lines, headers, and empty lines
            if not line or line.startswith('WEBVTT') or line.startswith('<?xml') or \
               '-->' in line or line.startswith('<') or line.isdigit():
                continue
            # Skip HTML tags
            if line.startswith('<') and line.endswith('>'):
                continue
            # Add text content
            if line:
                text_parts.append(line)
        
        return ' '.join(text_parts).strip()
    
    async def _process_video_entry(self, url: str, info: Dict[str, Any], data_dir: str, ydl: Any,
                                   playlist_title: Optional[str] = None, playlist_url: Optional[str] = None) -> Dict[str, Any]:
        if self._is_shorts_video(url, info):
            logger.info(f"⏭️ Skipping YouTube Short: {url} (duration: {info.get('duration', 'unknown')}s)")
            return {
                "filtered": True,
                "filter_reason": "youtube_short",
                "download_method": "youtube",
                "source_url": url,
                "playlist_title": playlist_title,
                "playlist_url": playlist_url
            }
        
        if self._is_sponsored_video(info):
            logger.info(f"⏭️ Skipping sponsored video: {url} (title: {info.get('title', 'unknown')})")
            return {
                "filtered": True,
                "filter_reason": "sponsored_content",
                "download_method": "youtube",
                "source_url": url,
                "playlist_title": playlist_title,
                "playlist_url": playlist_url
            }
        
        transcript_text = self._extract_transcript(info, ydl)
        if not transcript_text:
            logger.info(f"⏭️ Skipping YouTube video (no transcript available): {url} (title: {info.get('title', 'unknown')})")
            return {
                "filtered": True,
                "filter_reason": "no_transcript",
                "download_method": "youtube",
                "source_url": url,
                "playlist_title": playlist_title,
                "playlist_url": playlist_url
            }
        
        filename = f"youtube_{hashlib.md5(url.encode()).hexdigest()[:8]}.txt"
        file_path = os.path.join(data_dir, filename)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(transcript_text)
        
        import json
        metadata = {
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "source_url": url,
            "content_type": "transcript",
            "has_transcript": True
        }
        if playlist_title:
            metadata["playlist_title"] = playlist_title
        if playlist_url:
            metadata["playlist_url"] = playlist_url
        
        metadata_filename = f"youtube_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
        metadata_path = os.path.join(data_dir, metadata_filename)
        
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
        
        return {
            "title": filename,
            "file_path": file_path,
            "content_type": "text/plain",
            "file_size": os.path.getsize(file_path),
            "download_method": "youtube",
            "downloaded_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "has_transcript": True,
            "source_url": url,
            "playlist_title": playlist_title,
            "playlist_url": playlist_url
        }
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Extract YouTube metadata and transcript (if available), excluding Shorts and sponsored content"""
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
            from yt_dlp.utils import ExtractorError
            
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True,
                "writesubtitles": False,
                "writeautomaticsub": False
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            # Handle playlists by iterating through entries
            if info.get("_type") == "playlist":
                playlist_title = info.get("title")
                entries = info.get("entries") or []
                processed_entries: List[Dict[str, Any]] = []
                for entry in entries:
                    video_url = entry.get("webpage_url") or entry.get("url")
                    if not video_url:
                        continue
                    video_info = entry
                    if entry.get("_type") in {"url", "url_transparent"} or not entry.get("duration"):
                        try:
                            video_info = ydl.extract_info(video_url, download=False)
                        except Exception as ex:
                            logger.warning(f"Failed to extract playlist entry {video_url}: {ex}")
                            processed_entries.append({
                                "filtered": True,
                                "filter_reason": "playlist_extract_error",
                                "download_method": "youtube",
                                "source_url": video_url,
                                "playlist_title": playlist_title,
                                "playlist_url": url
                            })
                            continue
                    result = await self._process_video_entry(video_url, video_info, data_dir, ydl,
                                                              playlist_title=playlist_title, playlist_url=url)
                    processed_entries.append(result)
                return {
                    "playlist": True,
                    "playlist_title": playlist_title,
                    "playlist_url": url,
                    "entries": processed_entries
                }
            
            # Single video handling
            return await self._process_video_entry(url, info, data_dir, ydl)
            
        except (DownloadError, ExtractorError) as e:
            logger.error(f"YouTube extraction error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting YouTube content: {e}")
            return None
