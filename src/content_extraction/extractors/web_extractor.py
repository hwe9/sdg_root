"""
General web scraping extractor for various websites
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio
from bs4 import BeautifulSoup, Comment
import re
from urllib.parse import urljoin, urlparse
from .base_extractor import BaseExtractor, ExtractedContent

logger = logging.getLogger(__name__)

class WebExtractor(BaseExtractor):
    """
    General purpose web content extractor
    Handles various website types with smart content detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.content_selectors = self._load_content_selectors()
        self.blocked_elements = ['script', 'style', 'nav', 'header', 'footer', 'aside']
    
    def _load_content_selectors(self) -> Dict[str, List[str]]:
        """Load CSS selectors for different website types"""
        return {
            "article": [
                "article", 
                ".article", 
                ".post", 
                ".content", 
                ".main-content",
                ".entry-content",
                "#content",
                ".page-content"
            ],
            "title": [
                "h1",
                ".title", 
                ".article-title", 
                ".post-title",
                "title"
            ],
            "summary": [
                ".summary", 
                ".excerpt", 
                ".abstract", 
                ".lead",
                ".intro"
            ],
            "metadata": [
                ".metadata", 
                ".post-meta", 
                ".article-meta",
                "time[datetime]",
                ".author",
                ".date"
            ]
        }
    
    def validate_source(self, source_url: str) -> bool:
        """Validate if URL is suitable for web scraping"""
        try:
            parsed = urlparse(source_url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Block certain file types
            blocked_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
            if any(source_url.lower().endswith(ext) for ext in blocked_extensions):
                return False
            
            return True
        except:
            return False
    
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract content from web page"""
        try:
            if not self.validate_source(source_url):
                logger.warning(f"Invalid source URL: {source_url}")
                return []
            
            # Fetch page content
            response = await self.fetch_with_retry(source_url)
            if not response:
                return []
            
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            self._clean_soup(soup)
            
            # Extract structured content
            extracted_content = self._extract_structured_content(soup, source_url)
            
            if extracted_content:
                # Enhance with additional processing
                extracted_content = await self._enhance_content(extracted_content, soup)
                return [extracted_content]
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting from {source_url}: {e}")
            return []
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Remove unwanted elements from soup"""
        # Remove blocked elements
        for element_name in self.blocked_elements:
            for element in soup.find_all(element_name):
                element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove elements with no text content
        for element in soup.find_all():
            if not element.get_text(strip=True):
                element.decompose()
    
    def _extract_structured_content(self, soup: BeautifulSoup, source_url: str) -> Optional[ExtractedContent]:
        """Extract structured content from soup"""
        # Extract title
        title = self._extract_title(soup)
        if not title:
            return None
        
        # Extract main content
        content = self._extract_main_content(soup)
        if not content or len(content.strip()) < 100:
            return None
        
        # Extract summary
        summary = self._extract_summary(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup, source_url)
        
        # Detect language and region
        language = self._detect_language(soup, content)
        region = self._detect_region_from_content(content, soup)
        
        extracted_content = ExtractedContent(
            title=title,
            content=content,
            summary=summary,
            url=source_url,
            source_type="web_scraping",
            language=language,
            region=region,
            metadata=metadata
        )
        
        # Calculate quality score
        extracted_content.quality_score = self.estimate_quality_score(extracted_content)
        
        return extracted_content
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        for selector in self.content_selectors["title"]:
            elements = soup.select(selector)
            if elements:
                title = elements[0].get_text(strip=True)
                if title and len(title) > 10:
                    return title[:200]  # Limit title length
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)[:200]
        
        return ""
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using multiple strategies"""
        content_parts = []
        
        # Strategy 1: Use content selectors
        for selector in self.content_selectors["article"]:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > 200:  # Minimum content length
                        content_parts.append(text)
                        break
                if content_parts:
                    break
        
        # Strategy 2: Find paragraphs with substantial content
        if not content_parts:
            paragraphs = soup.find_all('p')
            paragraph_texts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Minimum paragraph length
                    paragraph_texts.append(text)
            
            if len(paragraph_texts) >= 3:  # At least 3 substantial paragraphs
                content_parts = paragraph_texts
        
        # Strategy 3: Extract all text as fallback
        if not content_parts:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text) > 500:
                    content_parts = [text]
        
        # Combine and clean content
        if content_parts:
            combined_content = '\n\n'.join(content_parts)
            # Clean up whitespace
            combined_content = re.sub(r'\s+', ' ', combined_content).strip()
            return combined_content
        
        return ""
    
    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract summary/excerpt"""
        for selector in self.content_selectors["summary"]:
            elements = soup.select(selector)
            if elements:
                summary = elements[0].get_text(strip=True)
                if summary and len(summary) > 50:
                    return summary[:500]  # Limit summary length
        
        # Fallback: first paragraph or meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'][:500]
        
        # First substantial paragraph
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 100:
                return text[:500]
        
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {
            "source_domain": urlparse(source_url).netloc,
            "extraction_method": "web_scraping"
        }
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                metadata[f"meta_{name}"] = content
        
        # Author
        author_selectors = ['.author', '[rel="author"]', '.byline']
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                metadata['author'] = author_elem.get_text(strip=True)
                break
        
        # Publication date
        date_selectors = ['time[datetime]', '.date', '.published']
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                if date_text:
                    metadata['publication_date'] = date_text
                    break
        
        # Word count
        content = soup.get_text()
        word_count = len(re.findall(r'\b\w+\b', content))
        metadata['word_count'] = word_count
        
        return metadata
    
    def _detect_language(self, soup: BeautifulSoup, content: str) -> str:
        """Detect content language"""
        # Check HTML lang attribute
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            lang = html_tag['lang'][:2]  # Get first 2 characters
            return lang
        
        # Simple language detection based on common words
        language_indicators = {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'for'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'in', 'zu'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être'],
            'es': ['el', 'de', 'que', 'y', 'a', 'en', 'un'],
            'zh': ['的', '是', '在', '了', '不', '与', '也'],
        }
        
        content_lower = content.lower()
        scores = {}
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in indicators if word in content_lower)
            scores[lang] = score
        
        if scores:
            detected_lang = max(scores, key=scores.get)
            if scores[detected_lang] > 2:  # Minimum threshold
                return detected_lang
        
        return 'en'  # Default to English
    
    def _detect_region_from_content(self, content: str, soup: BeautifulSoup) -> str:
        """Detect region from content and metadata"""
        # Check meta tags first
        geo_meta = soup.find('meta', attrs={'name': 'geo.region'})
        if geo_meta:
            return geo_meta.get('content', '')
        
        # Region keywords detection
        region_keywords = {
            "EU": ["european union", "europe", "eu", "brussels", "eurostat"],
            "USA": ["united states", "america", "usa", "washington dc"],
            "China": ["china", "chinese", "beijing", "prc"],
            "India": ["india", "indian", "new delhi", "bharat"],
            "ASEAN": ["asean", "southeast asia", "southeast asian"],
            "BRICS": ["brics", "emerging economies"]
        }
        
        content_lower = content.lower()
        region_scores = {}
        
        for region, keywords in region_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                region_scores[region] = score
        
        if region_scores:
            return max(region_scores, key=region_scores.get)
        
        return ""
    
    async def _enhance_content(self, content: ExtractedContent, soup: BeautifulSoup) -> ExtractedContent:
        """Enhance extracted content with additional processing"""
        # Extract links for further processing
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith(('http://', 'https://')):
                links.append({
                    'url': href,
                    'text': link.get_text(strip=True)
                })
        
        content.metadata['internal_links'] = links[:20]  # Limit links
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                'src': img['src'],
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        content.metadata['images'] = images[:10]  # Limit images
        
        # Simple SDG keyword detection for initial relevance
        content.sdg_relevance = self._detect_sdg_relevance(content.content)
        
        return content
    
    def _detect_sdg_relevance(self, content: str) -> List[int]:
        """Simple SDG relevance detection"""
        # Basic keyword-based SDG detection
        sdg_keywords = {
            1: ["poverty", "poor", "income"],
            2: ["hunger", "food", "agriculture"],
            3: ["health", "medical", "disease"],
            13: ["climate", "carbon", "emission", "greenhouse"],
            # Add more as needed
        }
        
        content_lower = content.lower()
        relevant_sdgs = []
        
        for sdg_id, keywords in sdg_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                relevant_sdgs.append(sdg_id)
        
        return relevant_sdgs
