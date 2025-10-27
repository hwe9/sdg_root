"""
RSS Feed content extractor
"""
import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
import asyncio
import feedparser
from datetime import datetime
import re
from .base_extractor import BaseExtractor
from .base_extractor import ExtractedContent

logger = logging.getLogger(__name__)

class RSSExtractor(BaseExtractor):
    """
    RSS/Atom feed content extractor
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_entries = config.get("max_entries_per_feed", 50)
    
    def validate_source(self, source_url: str) -> bool:
        """Validate RSS feed URL"""
        rss_indicators = ['/rss', '/feed', '.rss', '.xml', 'rss.xml', 'feed.xml']
        return any(indicator in source_url.lower() for indicator in rss_indicators)
    
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract content from RSS feed"""
        try:
            # Fetch RSS feed
            response = await self.fetch_with_retry(source_url)
            if not response:
                return []
            
            feed_content = await response.text()
            
            # Parse feed
            feed = feedparser.parse(feed_content)
            
            if feed.bozo and hasattr(feed, 'bozo_exception'):
                logger.warning(f"RSS parsing issues for {source_url}: {feed.bozo_exception}")
            
            extracted_items = []
            
            # Process feed entries
            for entry in feed.entries[:self.max_entries]:
                content_item = self._process_feed_entry(entry, feed, source_url)
                if content_item:
                    extracted_items.append(content_item)
            
            logger.info(f"Extracted {len(extracted_items)} items from RSS feed: {source_url}")
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error processing RSS feed {source_url}: {e}")
            return []
    
    def _process_feed_entry(self, entry: Any, feed: Any, source_url: str) -> Optional[ExtractedContent]:
        """Process individual RSS feed entry"""
        try:
            # Extract title
            title = getattr(entry, 'title', '').strip()
            if not title:
                return None
            
            # Extract content
            content = self._extract_entry_content(entry)
            if not content or len(content.strip()) < 50:
                return None
            
            # Extract summary
            summary = getattr(entry, 'summary', '')
            if not summary:
                summary = content[:300] + "..." if len(content) > 300 else content
            
            # Extract URL
            entry_url = getattr(entry, 'link', source_url)
            
            # Extract publication date
            pub_date = self._extract_publication_date(entry)
            
            # Extract metadata
            metadata = self._extract_entry_metadata(entry, feed)
            
            # Detect language and region
            language = self._detect_entry_language(entry, feed)
            region = self._detect_entry_region(content, metadata)
            
            content_item = ExtractedContent(
                title=title[:200],  # Limit title length
                content=content,
                summary=summary[:500],  # Limit summary length
                url=entry_url,
                source_type="rss_feed",
                language=language,
                region=region,
                extracted_at=datetime.utcnow(),
                metadata=metadata
            )
            
            # Calculate quality score
            content_item.quality_score = self._calculate_rss_quality_score(content_item, entry)
            
            # Simple SDG relevance detection
            content_item.sdg_relevance = self._detect_sdg_keywords(content)
            
            return content_item
            
        except Exception as e:
            logger.error(f"Error processing RSS entry: {e}")
            return None
    
    def _extract_entry_content(self, entry: Any) -> str:
        """Extract content from RSS entry"""
        # Try different content fields
        content_fields = ['content', 'description', 'summary']
        
        for field in content_fields:
            content_data = getattr(entry, field, None)
            if content_data:
                if isinstance(content_data, list) and content_data:
                    # Handle content list (like Atom feeds)
                    content_text = content_data[0].get('value', '')
                elif isinstance(content_data, dict):
                    content_text = content_data.get('value', '')
                else:
                    content_text = str(content_data)
                
                if content_text:
                    # Clean HTML tags
                    clean_content = re.sub(r'<[^>]+>', ' ', content_text)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    
                    if len(clean_content) > 100:
                        return clean_content
        
        return ""
    
    def _extract_publication_date(self, entry: Any) -> Optional[str]:
        """Extract publication date from entry"""
        date_fields = ['published_parsed', 'updated_parsed']
        
        for field in date_fields:
            date_data = getattr(entry, field, None)
            if date_data:
                try:
                    # Convert time struct to ISO format
                    dt = datetime(*date_data[:6])
                    return dt.isoformat()
                except:
                    continue
        
        # Try string fields as fallback
        string_date_fields = ['published', 'updated']
        for field in string_date_fields:
            date_str = getattr(entry, field, '')
            if date_str:
                return date_str
        
        return None
    
    def _extract_entry_metadata(self, entry: Any, feed: Any) -> Dict[str, Any]:
        """Extract metadata from RSS entry and feed"""
        metadata = {
            "feed_title": getattr(feed.feed, 'title', ''),
            "feed_description": getattr(feed.feed, 'description', ''),
            "feed_url": getattr(feed.feed, 'link', ''),
            "entry_id": getattr(entry, 'id', ''),
            "extraction_method": "rss_feed"
        }
        
        # Author information
        author = getattr(entry, 'author', '')
        if author:
            metadata['author'] = author
        
        # Tags/categories
        tags = getattr(entry, 'tags', [])
        if tags:
            tag_list = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_list.append(tag.get('term', ''))
                else:
                    tag_list.append(str(tag))
            metadata['tags'] = [t for t in tag_list if t]
        
        # Publication date
        pub_date = self._extract_publication_date(entry)
        if pub_date:
            metadata['publication_date'] = pub_date
        
        # Additional fields
        for field in ['rights', 'publisher']:
            value = getattr(entry, field, '')
            if value:
                metadata[field] = value
        
        return metadata
    
    def _detect_entry_language(self, entry: Any, feed: Any) -> str:
        """Detect language from entry or feed"""
        # Check entry language
        entry_lang = getattr(entry, 'language', '')
        if entry_lang:
            return entry_lang[:2]
        
        # Check feed language
        feed_lang = getattr(feed.feed, 'language', '')
        if feed_lang:
            return feed_lang[:2]
        
        # Simple detection based on content
        title = getattr(entry, 'title', '')
        content = self._extract_entry_content(entry)
        combined_text = f"{title} {content}".lower()
        
        # Basic language indicators
        if any(word in combined_text for word in ['the', 'and', 'is', 'in']):
            return 'en'
        elif any(word in combined_text for word in ['der', 'die', 'das', 'und']):
            return 'de'
        elif any(word in combined_text for word in ['le', 'de', 'et', 'dans']):
            return 'fr'
        
        return 'en'  # Default
    
    def _detect_entry_region(self, content: str, metadata: Dict[str, Any]) -> str:
        """Detect region from content and metadata"""
        # Check feed metadata first
        feed_url = metadata.get('feed_url', '')
        if feed_url:
            # Domain-based region detection
            domain_regions = {
                '.eu': 'EU',
                '.gov': 'USA',
                '.cn': 'China',
                '.in': 'India'
            }
            for domain, region in domain_regions.items():
                if domain in feed_url:
                    return region
        
        # Content-based detection
        region_indicators = {
            'EU': ['european', 'europe', 'brussels', 'eu'],
            'USA': ['america', 'united states', 'washington'],
            'China': ['china', 'chinese', 'beijing'],
            'India': ['india', 'indian', 'delhi']
        }
        
        content_lower = content.lower()
        for region, keywords in region_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                return region
        
        return ""
    
    def _calculate_rss_quality_score(self, content: ExtractedContent, entry: Any) -> float:
        """Calculate quality score for RSS content"""
        score = 0.0
        
        # Title quality (0-0.2)
        if content.title and len(content.title.strip()) > 20:
            score += 0.2
        elif content.title and len(content.title.strip()) > 10:
            score += 0.1
        
        # Content length (0-0.3)
        content_length = len(content.content)
        if content_length > 1000:
            score += 0.3
        elif content_length > 500:
            score += 0.2
        elif content_length > 200:
            score += 0.1
        
        # Has author (0-0.1)
        if content.metadata.get('author'):
            score += 0.1
        
        # Has publication date (0-0.1)
        if content.metadata.get('publication_date'):
            score += 0.1
        
        # Has tags/categories (0-0.1)
        if content.metadata.get('tags'):
            score += 0.1
        
        # Has valid URL (0-0.1)
        if content.url and content.url.startswith(('http://', 'https://')):
            score += 0.1
        
        # SDG relevance (0-0.1)
        if content.sdg_relevance:
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_sdg_keywords(self, content: str) -> List[int]:
        """Simple SDG keyword detection"""
        sdg_patterns = {
            1: r'\b(poverty|poor|income|wealth|social protection)\b',
            2: r'\b(hunger|food|nutrition|agriculture|farming)\b',
            3: r'\b(health|medical|disease|mortality|healthcare)\b',
            4: r'\b(education|learning|school|literacy|skills)\b',
            5: r'\b(gender|women|girls|equality|empowerment)\b',
            6: r'\b(water|sanitation|hygiene|drinking water)\b',
            7: r'\b(energy|renewable|electricity|clean energy)\b',
            8: r'\b(employment|jobs|economic growth|decent work)\b',
            9: r'\b(infrastructure|innovation|industry|technology)\b',
            10: r'\b(inequality|inclusion|discrimination|equity)\b',
            11: r'\b(cities|urban|housing|transport|sustainable cities)\b',
            12: r'\b(consumption|production|waste|recycling|sustainable)\b',
            13: r'\b(climate|carbon|emission|greenhouse|adaptation)\b',
            14: r'\b(ocean|marine|sea|fisheries|aquatic)\b',
            15: r'\b(forest|biodiversity|ecosystem|wildlife|conservation)\b',
            16: r'\b(peace|justice|institutions|governance|rule of law)\b',
            17: r'\b(partnership|cooperation|global|development finance)\b'
        }
        
        content_lower = content.lower()
        relevant_sdgs = []
        
        for sdg_id, pattern in sdg_patterns.items():
            if re.search(pattern, content_lower):
                relevant_sdgs.append(sdg_id)
        
        return relevant_sdgs
    
    async def extract_multiple_feeds(self, feed_urls: List[str], **kwargs) -> List[ExtractedContent]:
        """Extract content from multiple RSS feeds concurrently"""
        all_content = []
        
        # Process feeds in batches to avoid overwhelming servers
        batch_size = self.config.get("concurrent_feeds", 3)
        
        for i in range(0, len(feed_urls), batch_size):
            batch_urls = feed_urls[i:i + batch_size]
            batch_results = await self.process_batch(batch_urls, **kwargs)
            all_content.extend(batch_results)
            
            # Brief pause between batches
            if i + batch_size < len(feed_urls):
                await asyncio.sleep(1)
        
        logger.info(f"Extracted {len(all_content)} total items from {len(feed_urls)} RSS feeds")
        return all_content
