"""
Feed Source Handler
Handles RSS/Atom feeds
"""
import logging
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import feedparser
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedSource:
    """Handler for RSS/Atom feed sources"""
    
    def __init__(self):
        pass
    
    async def initialize(self):
        """Initialize feed source handler"""
        logger.info("✅ Feed source handler initialized")
    
    async def download(self, url: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Download and parse feed content"""
        try:
            # Parse feed
            feed = feedparser.parse(url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing issues for {url}: {feed.bozo_exception}")
            
            # Extract feed metadata
            feed_metadata = {
                "title": feed.feed.get("title", "Unknown Feed"),
                "description": feed.feed.get("description", ""),
                "link": feed.feed.get("link", ""),
                "entries": []
            }
            
            # Process entries
            for entry in feed.entries[:10]:  # Limit to first 10 entries
                entry_data = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.term for tag in entry.get("tags", [])],
                    "content": self._extract_content(entry)
                }
                feed_metadata["entries"].append(entry_data)
            
            # Save to file
            filename = f"feed_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
            file_path = os.path.join(data_dir, filename)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(feed_metadata, ensure_ascii=False, indent=2))
            
            return {
                "title": filename,
                "file_path": file_path,
                "content_type": "application/json",
                "file_size": os.path.getsize(file_path),
                "download_method": "feed",
                "downloaded_at": datetime.utcnow().isoformat(),
                "metadata": feed_metadata
            }
            
        except Exception as e:
            logger.error(f"Error downloading feed {url}: {e}")
            return None
    
    def _extract_content(self, entry) -> str:
        """Extract content from feed entry"""
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            return entry.content[0].value if entry.content else ""
        elif hasattr(entry, 'summary') and entry.summary:
            return entry.summary
        elif hasattr(entry, 'description') and entry.description:
            return entry.description
        else:
            return ""
