"""
Newsletter content extractor
Handles email newsletters and similar formatted content
"""
import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
import re
from bs4 import BeautifulSoup
from .base_extractor import BaseExtractor
from .base_extractor import ExtractedContent
from .web_extractor import WebExtractor

logger = logging.getLogger(__name__)

class NewsletterExtractor(WebExtractor):
    """
    Newsletter content extractor
    Extends WebExtractor with newsletter-specific logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.newsletter_patterns = self._load_newsletter_patterns()
    
    def _load_newsletter_patterns(self) -> Dict[str, List[str]]:
        """Load newsletter-specific patterns"""
        return {
            "newsletter_indicators": [
                "newsletter",
                "weekly update",
                "monthly digest",
                "news digest",
                "bulletin",
                "briefing"
            ],
            "content_sections": [
                ".newsletter-content",
                ".email-content",
                ".digest-content", 
                ".bulletin-content",
                "table[role='presentation']",
                "td.content"
            ],
            "article_sections": [
                ".article-item",
                ".news-item",
                ".digest-item",
                ".story",
                "tr.article",
                "div[style*='border']"
            ]
        }
    
    def validate_source(self, source_url: str) -> bool:
        """Validate if source is newsletter-like"""
        newsletter_indicators = self.newsletter_patterns["newsletter_indicators"]
        url_lower = source_url.lower()
        
        # Check URL for newsletter indicators
        if any(indicator in url_lower for indicator in newsletter_indicators):
            return True
        
        # Also validate as general web content
        return super().validate_source(source_url)
    
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract newsletter content with multi-article support"""
        try:
            # Use parent class to fetch and clean content
            response = await self.fetch_with_retry(source_url)
            if not response:
                return []
            
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check if this is actually a newsletter
            if not self._is_newsletter_content(soup, source_url):
                # Fall back to regular web extraction
                return await super().extract(source_url, **kwargs)
            
            # Clean soup
            self._clean_soup(soup)
            
            # Extract multiple articles from newsletter
            articles = self._extract_newsletter_articles(soup, source_url)
            
            if not articles:
                # Fall back to single content extraction
                single_content = self._extract_structured_content(soup, source_url)
                if single_content:
                    single_content.source_type = "newsletter"
                    return [single_content]
            
            return articles
            
        except Exception as e:
            logger.error(f"Error extracting newsletter from {source_url}: {e}")
            return []
    
    def _is_newsletter_content(self, soup: BeautifulSoup, source_url: str) -> bool:
        """Determine if content is newsletter-like"""
        # Check URL
        url_lower = source_url.lower()
        newsletter_indicators = self.newsletter_patterns["newsletter_indicators"]
        if any(indicator in url_lower for indicator in newsletter_indicators):
            return True
        
        # Check page content
        page_text = soup.get_text().lower()
        if any(indicator in page_text[:500] for indicator in newsletter_indicators):
            return True
        
        # Check for newsletter-like structure
        # Multiple article-like sections
        article_sections = soup.select('.article, .story, .news-item, .digest-item')
        if len(article_sections) >= 3:
            return True
        
        # Email-like table structure
        if soup.find('table', {'role': 'presentation'}):
            return True
        
        # Newsletter-specific meta tags
        newsletter_meta = soup.find('meta', attrs={
            'name': lambda x: x and 'newsletter' in x.lower() if x else False
        })
        if newsletter_meta:
            return True
        
        return False
    
    def _extract_newsletter_articles(self, soup: BeautifulSoup, source_url: str) -> List[ExtractedContent]:
        """Extract individual articles from newsletter"""
        articles = []
        
        # Try different strategies to find article sections
        article_sections = self._find_article_sections(soup)
        
        newsletter_metadata = self._extract_newsletter_metadata(soup, source_url)
        
        for i, section in enumerate(article_sections):
            article = self._extract_newsletter_article(section, source_url, i, newsletter_metadata)
            if article:
                articles.append(article)
        
        return articles
    
    def _find_article_sections(self, soup: BeautifulSoup) -> List:
        """Find individual article sections in newsletter"""
        sections = []
        
        # Strategy 1: Use newsletter-specific selectors
        for selector in self.newsletter_patterns["article_sections"]:
            found_sections = soup.select(selector)
            if found_sections and len(found_sections) >= 2:
                return found_sections
        
        # Strategy 2: Find repeating patterns
        # Look for multiple divs/sections with similar structure
        potential_sections = []
        
        # Try divs with similar classes
        all_divs = soup.find_all('div', class_=True)
        class_groups = {}
        
        for div in all_divs:
            classes = ' '.join(div.get('class', []))
            if classes:
                class_groups.setdefault(classes, []).append(div)
        
        # Find groups with multiple similar elements
        for class_name, divs in class_groups.items():
            if len(divs) >= 3:  # At least 3 similar sections
                # Check if they contain substantial content
                content_divs = []
                for div in divs:
                    text = div.get_text(strip=True)
                    if len(text) > 100:  # Minimum content length
                        content_divs.append(div)
                
                if len(content_divs) >= 2:
                    potential_sections = content_divs
                    break
        
        # Strategy 3: Table rows (email newsletters often use tables)
        if not potential_sections:
            table_rows = soup.find_all('tr')
            content_rows = []
            
            for row in table_rows:
                text = row.get_text(strip=True)
                if len(text) > 100 and not self._is_header_footer_row(row):
                    content_rows.append(row)
            
            if len(content_rows) >= 2:
                potential_sections = content_rows
        
        # Strategy 4: Paragraphs with headers
        if not potential_sections:
            # Find h2/h3 elements that might be article headers
            headers = soup.find_all(['h2', 'h3', 'h4'])
            for header in headers:
                # Get content after header until next header
                article_content = []
                current = header.next_sibling
                
                while current:
                    if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4']:
                        break
                    if hasattr(current, 'get_text'):
                        text = current.get_text(strip=True)
                        if text:
                            article_content.append(current)
                    current = current.next_sibling
                
                if article_content:
                    # Create a wrapper for this article section
                    wrapper = soup.new_tag('div')
                    wrapper.append(header)
                    for content in article_content:
                        wrapper.append(content)
                    potential_sections.append(wrapper)
        
        return potential_sections[:20]  # Limit number of sections
    
    def _is_header_footer_row(self, row) -> bool:
        """Check if table row is header/footer content"""
        text = row.get_text().lower()
        header_footer_indicators = [
            'unsubscribe', 'privacy policy', 'terms of service',
            'follow us', 'social media', 'contact us', 'newsletter',
            'view in browser', 'email preferences', 'copyright'
        ]
        
        return any(indicator in text for indicator in header_footer_indicators)
    
    def _extract_newsletter_article(self, section, source_url: str, index: int, 
                                  newsletter_metadata: Dict[str, Any]) -> Optional[ExtractedContent]:
        """Extract individual article from newsletter section"""
        try:
            # Extract title
            title = self._extract_section_title(section)
            if not title:
                title = f"Newsletter Article {index + 1}"
            
            # Extract content
            content = section.get_text(separator=' ', strip=True)
            if len(content) < 100:  # Minimum content length
                return None
            
            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Extract links
            links = self._extract_section_links(section, source_url)
            
            # Generate article URL (use first link or base URL with anchor)
            article_url = source_url
            if links:
                article_url = links[0]['url']
            else:
                article_url = f"{source_url}#article-{index + 1}"
            
            # Create summary (first paragraph or truncated content)
            summary = self._extract_section_summary(section, content)
            
            # Combine newsletter metadata with article-specific metadata
            article_metadata = {
                **newsletter_metadata,
                "article_index": index + 1,
                "links": links,
                "section_type": "newsletter_article"
            }
            
            # Detect SDG relevance
            sdg_relevance = self._detect_sdg_keywords_newsletter(content + " " + title)
            
            article = ExtractedContent(
                title=title,
                content=content,
                summary=summary,
                url=article_url,
                source_type="newsletter",
                language=newsletter_metadata.get('language', 'en'),
                region=newsletter_metadata.get('region', ''),
                metadata=article_metadata,
                sdg_relevance=sdg_relevance
            )
            
            # Calculate quality score
            article.quality_score = self._calculate_newsletter_quality_score(article, section)
            
            return article
            
        except Exception as e:
            logger.error(f"Error extracting newsletter article {index}: {e}")
            return None
    
    def _extract_section_title(self, section) -> str:
        """Extract title from newsletter section"""
        # Look for header tags
        for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5']:
            header = section.find(header_tag)
            if header:
                title = header.get_text(strip=True)
                if len(title) > 5:
                    return title[:200]
        
        # Look for bold/strong text at beginning
        bold_elements = section.find_all(['b', 'strong'])
        for bold in bold_elements:
            text = bold.get_text(strip=True)
            if len(text) > 10 and len(text) < 100:
                return text
        
        # Look for first sentence
        text = section.get_text(strip=True)
        sentences = re.split(r'[.!?]', text)
        if sentences and len(sentences[0]) > 10:
            return sentences[0][:200]
        
        return ""
    
    def _extract_section_links(self, section, base_url: str) -> List[Dict[str, str]]:
        """Extract links from newsletter section"""
        links = []
        for link in section.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                from urllib.parse import urljoin
                from urllib.parse import urlparse
                base_domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
                href = urljoin(base_domain, href)
            elif not href.startswith(('http://', 'https://')):
                continue
            
            if text and href:
                links.append({
                    'url': href,
                    'text': text,
                    'type': 'article_link'
                })
        
        return links[:5]  # Limit links per article
    
    def _extract_section_summary(self, section, content: str) -> str:
        """Extract or generate summary for newsletter section"""
        # Look for explicit summary/excerpt
        summary_selectors = ['.summary', '.excerpt', '.intro', '.lead']
        for selector in summary_selectors:
            summary_elem = section.select_one(selector)
            if summary_elem:
                summary = summary_elem.get_text(strip=True)
                if len(summary) > 50:
                    return summary[:400]
        
        # Use first paragraph if available
        paragraphs = section.find_all('p')
        if paragraphs:
            first_p = paragraphs[0].get_text(strip=True)
            if len(first_p) > 50:
                return first_p[:400]
        
        # Fallback to truncated content
        return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_newsletter_metadata(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Extract newsletter-level metadata"""
        metadata = {
            "source_type": "newsletter",
            "extraction_method": "newsletter_extractor"
        }
        
        # Newsletter title
        title_elem = soup.find('title')
        if title_elem:
            metadata['newsletter_title'] = title_elem.get_text(strip=True)
        
        # Newsletter date
        date_selectors = [
            'meta[name="date"]',
            'meta[property="article:published_time"]',
            '.newsletter-date',
            '.issue-date'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_value = date_elem.get('content') or date_elem.get_text(strip=True)
                if date_value:
                    metadata['newsletter_date'] = date_value
                    break
        
        # Newsletter organization/sender
        org_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.newsletter-from',
            '.sender'
        ]
        
        for selector in org_selectors:
            org_elem = soup.select_one(selector)
            if org_elem:
                org_value = org_elem.get('content') or org_elem.get_text(strip=True)
                if org_value:
                    metadata['newsletter_organization'] = org_value
                    break
        
        # Language detection
        html_elem = soup.find('html')
        if html_elem and html_elem.get('lang'):
            metadata['language'] = html_elem['lang'][:2]
        else:
            metadata['language'] = 'en'
        
        # Region detection (basic)
        content_text = soup.get_text()
        metadata['region'] = self._detect_region_from_content(content_text, soup)
        
        return metadata
    
    def _detect_sdg_keywords_newsletter(self, text: str) -> List[int]:
        """Enhanced SDG keyword detection for newsletters"""
        # Newsletter content often has more context, so use enhanced patterns
        enhanced_sdg_patterns = {
            1: r'\b(poverty|poor|low.income|wealth.gap|social.protection|basic.needs)\b',
            2: r'\b(hunger|food.security|malnutrition|agriculture|farming|crop)\b',
            3: r'\b(health|healthcare|medical|disease|mortality|wellbeing|pandemic)\b',
            4: r'\b(education|school|learning|literacy|skills|training|university)\b',
            5: r'\b(gender|women|girls|equality|empowerment|discrimination)\b',
            6: r'\b(water|sanitation|hygiene|drinking.water|clean.water)\b',
            7: r'\b(energy|renewable|solar|wind|electricity|clean.energy|fossil)\b',
            8: r'\b(employment|jobs|economic.growth|decent.work|unemployment)\b',
            9: r'\b(infrastructure|innovation|industry|technology|research|development)\b',
            10: r'\b(inequality|inclusion|discrimination|equity|marginalized)\b',
            11: r'\b(cities|urban|housing|transport|sustainable.cities|smart.city)\b',
            12: r'\b(consumption|production|waste|recycling|sustainable|circular.economy)\b',
            13: r'\b(climate|carbon|emission|greenhouse|global.warming|adaptation)\b',
            14: r'\b(ocean|marine|sea|fisheries|aquatic|coral|plastic.pollution)\b',
            15: r'\b(forest|biodiversity|ecosystem|wildlife|conservation|deforestation)\b',
            16: r'\b(peace|justice|institutions|governance|rule.of.law|corruption)\b',
            17: r'\b(partnership|cooperation|global|development.finance|aid)\b'
        }
        
        text_lower = text.lower()
        text_lower = re.sub(r'[^\w\s]', ' ', text_lower)  # Remove punctuation
        relevant_sdgs = []
        
        for sdg_id, pattern in enhanced_sdg_patterns.items():
            if re.search(pattern, text_lower):
                relevant_sdgs.append(sdg_id)
        
        return relevant_sdgs
    
    def _calculate_newsletter_quality_score(self, article: ExtractedContent, section) -> float:
        """Calculate quality score for newsletter article"""
        score = 0.0
        
        # Base quality score from parent
        base_score = self.estimate_quality_score(article)
        score += base_score * 0.6
        
        # Newsletter-specific quality factors
        
        # Has links (0-0.1)
        links = article.metadata.get('links', [])
        if links:
            score += 0.1
        
        # Content structure (0-0.1)
        # Check if section has proper HTML structure
        if section.find(['p', 'div', 'span']):
            score += 0.1
        
        # Newsletter organization info (0-0.1)
        if article.metadata.get('newsletter_organization'):
            score += 0.1
        
        # SDG relevance bonus (0-0.1)
        if len(article.sdg_relevance) >= 2:
            score += 0.1
        elif len(article.sdg_relevance) >= 1:
            score += 0.05
        
        return min(score, 1.0)
