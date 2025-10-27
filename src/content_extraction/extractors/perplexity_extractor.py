"""
Perplexity.ai HTML Result Extractor (Non-API)
Parses Perplexity search results and extracts content with source citations
"""
import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
import re
import asyncio
from bs4 import BeautifulSoup
from bs4 import Comment
from urllib.parse import urljoin
from urllib.parse import urlparse
from datetime import datetime

from .base_extractor import BaseExtractor
from .base_extractor import ExtractedContent

logger = logging.getLogger(__name__)

class PerplexityExtractor(BaseExtractor):
    """
    Perplexity.ai search result extractor
    Parses Perplexity response pages and extracts answers with source citations
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.perplexity_selectors = self._load_perplexity_selectors()
        self.citation_patterns = self._load_citation_patterns()
    
    def _load_perplexity_selectors(self) -> Dict[str, List[str]]:
        """Load Perplexity-specific HTML selectors"""
        return {
            "answer_containers": [
                ".answer-content",
                ".response-text", 
                "[data-testid='answer']",
                ".prose-content",
                ".main-answer"
            ],
            "query_containers": [
                ".query-text",
                ".search-query",
                "[data-testid='query']",
                ".user-question"
            ],
            "source_citations": [
                ".citation",
                ".source-link",
                "[data-citation-index]",
                ".reference",
                "sup a"
            ],
            "source_list": [
                ".sources-list",
                ".references-section",
                ".source-references",
                "[data-testid='sources']"
            ],
            "follow_up_questions": [
                ".follow-up-questions",
                ".related-queries",
                ".suggested-questions"
            ]
        }
    
    def _load_citation_patterns(self) -> Dict[str, str]:
        """Load patterns for identifying citations"""
        return {
            "numbered_citation": r'\[(\d+)\]',
            "superscript_citation": r'<sup[^>]*>(\d+)</sup>',
            "parenthetical_citation": r'\((\d+)\)',
            "source_indicator": r'(source|according to|based on|from):\s*(.+?)(?=\.|$)',
        }
    
    def validate_source(self, source_url: str) -> bool:
        """Validate if URL is from Perplexity"""
        perplexity_domains = [
            'perplexity.ai',
            'www.perplexity.ai'
        ]
        
        parsed = urlparse(source_url.lower())
        return any(domain in parsed.netloc for domain in perplexity_domains)
    
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract content from Perplexity search result page"""
        try:
            response = await self.fetch_with_retry(source_url)
            if not response:
                return []
            
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Clean soup
            self._clean_soup(soup)
            
            # Extract query context
            user_query = self._extract_user_query(soup)
            
            # Extract main answer
            answer_content = self._extract_main_answer(soup, source_url)
            
            # Extract source citations and references
            citations = self._extract_citations(soup, source_url)
            
            # Extract follow-up questions
            follow_up_questions = self._extract_follow_up_questions(soup)
            
            if not answer_content:
                return []
            
            # Create extracted content
            extracted_content = ExtractedContent(
                title=self._generate_title(user_query, answer_content['text']),
                content=answer_content['text'],
                summary=self._create_summary(answer_content['text']),
                url=source_url,
                source_type="perplexity_search",
                language=kwargs.get('language', 'en'),
                region=kwargs.get('region', ''),
                metadata={
                    'user_query': user_query,
                    'source_citations': citations,
                    'follow_up_questions': follow_up_questions,
                    'extraction_method': 'perplexity_html_parser',
                    'answer_structure': answer_content.get('structure_type', 'prose'),
                    'citation_count': len(citations),
                    'high_quality_sources': self._count_high_quality_sources(citations)
                }
            )
            
            # Detect SDG relevance
            extracted_content.sdg_relevance = self._detect_sdg_relevance(
                answer_content['text'] + ' ' + user_query
            )
            
            # Calculate quality score
            extracted_content.quality_score = self._calculate_perplexity_quality_score(
                extracted_content, answer_content, citations
            )
            
            return [extracted_content]
            
        except Exception as e:
            logger.error(f"Error extracting Perplexity content from {source_url}: {e}")
            return []
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Remove unwanted elements"""
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _extract_user_query(self, soup: BeautifulSoup) -> str:
        """Extract the user's search query"""
        # Try different selectors for query
        for selector in self.perplexity_selectors["query_containers"]:
            query_elem = soup.select_one(selector)
            if query_elem:
                query_text = query_elem.get_text(strip=True)
                if len(query_text) > 5:
                    return query_text
        
        # Fallback: look for query in page title or meta tags
        title_elem = soup.find('title')
        if title_elem:
            title_text = title_elem.get_text().strip()
            # Remove "Perplexity" branding
            title_text = re.sub(r'\s*-\s*Perplexity.*$', '', title_text, flags=re.IGNORECASE)
            if len(title_text) > 10:
                return title_text
        
        return ""
    
    def _extract_main_answer(self, soup: BeautifulSoup, source_url: str) -> Optional[Dict[str, Any]]:
        """Extract the main answer content"""
        # Try different selectors for answer content
        answer_elem = None
        for selector in self.perplexity_selectors["answer_containers"]:
            found_elem = soup.select_one(selector)
            if found_elem:
                answer_elem = found_elem
                break
        
        if not answer_elem:
            # Fallback: look for largest text block
            text_blocks = soup.find_all(['div', 'p', 'article'], string=True)
            if text_blocks:
                # Find the longest text block
                answer_elem = max(text_blocks, key=lambda x: len(x.get_text(strip=True)))
        
        if not answer_elem:
            return None
        
        # Extract text content
        answer_text = answer_elem.get_text(separator=' ', strip=True)
        
        if len(answer_text) < 50:
            return None
        
        # Analyze structure
        structure_type = self._analyze_answer_structure(answer_elem)
        
        # Extract any embedded lists or structured data
        lists = answer_elem.find_all(['ul', 'ol'])
        tables = answer_elem.find_all('table')
        
        return {
            'text': answer_text,
            'structure_type': structure_type,
            'has_lists': len(lists) > 0,
            'has_tables': len(tables) > 0,
            'word_count': len(answer_text.split())
        }
    
    def _analyze_answer_structure(self, answer_elem) -> str:
        """Analyze the structure of the answer"""
        # Check for lists
        if answer_elem.find_all(['ul', 'ol']):
            return "structured_list"
        
        # Check for tables
        if answer_elem.find_all('table'):
            return "tabular_data"
        
        # Check for multiple paragraphs
        paragraphs = answer_elem.find_all('p')
        if len(paragraphs) > 2:
            return "multi_paragraph"
        
        return "prose"
    
    def _extract_citations(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract source citations and references"""
        citations = []
        
        # Method 1: Find citation links within text
        citation_links = soup.select('a[href*="source"], sup a, [data-citation] a')
        for link in citation_links:
            href = link.get('href')
            if href:
                citation_data = self._process_citation_link(link, href, base_url)
                if citation_data:
                    citations.append(citation_data)
        
        # Method 2: Find sources section
        for selector in self.perplexity_selectors["source_list"]:
            sources_section = soup.select_one(selector)
            if sources_section:
                source_links = sources_section.find_all('a', href=True)
                for link in source_links:
                    href = link['href']
                    citation_data = self._process_citation_link(link, href, base_url)
                    if citation_data:
                        citations.append(citation_data)
        
        # Method 3: Look for numbered references
        numbered_refs = soup.find_all(string=re.compile(r'\[\d+\]'))
        for ref in numbered_refs:
            # Try to find associated link
            parent = ref.parent if ref.parent else None
            if parent:
                link = parent.find('a', href=True)
                if link:
                    citation_data = self._process_citation_link(link, link['href'], base_url)
                    if citation_data:
                        citations.append(citation_data)
        
        # Remove duplicates and sort by relevance
        seen_urls = set()
        unique_citations = []
        for citation in citations:
            if citation['url'] not in seen_urls:
                seen_urls.add(citation['url'])
                unique_citations.append(citation)
        
        # Sort by quality score
        unique_citations.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return unique_citations[:15]  # Limit to most relevant citations
    
    def _process_citation_link(self, link_elem, href: str, base_url: str) -> Optional[Dict[str, Any]]:
        """Process individual citation link"""
        # Skip invalid links
        if not href or href.startswith(('javascript:', '#', 'mailto:')):
            return None
        
        # Convert relative URLs
        if href.startswith('/'):
            href = urljoin(base_url, href)
        elif not href.startswith(('http://', 'https://')):
            return None
        
        # Extract link text and title
        link_text = link_elem.get_text(strip=True)
        link_title = link_elem.get('title', '')
        
        # Get domain information
        parsed_url = urlparse(href)
        domain = parsed_url.netloc
        
        # Assess source quality
        quality_score = self._assess_source_quality(href, link_text, domain)
        
        # Categorize source type
        source_type = self._categorize_source(href, link_text, domain)
        
        # Check for download potential
        is_download = self._is_potential_download(href, link_text)
        
        return {
            'url': href,
            'text': link_text,
            'title': link_title,
            'domain': domain,
            'source_type': source_type,
            'quality_score': quality_score,
            'is_download': is_download,
            'sdg_relevance': self._assess_citation_sdg_relevance(href, link_text)
        }
    
    def _assess_source_quality(self, url: str, text: str, domain: str) -> float:
        """Assess the quality of a citation source"""
        score = 0.0
        
        # High-quality domains
        quality_domains = [
            'un.org', 'who.int', 'worldbank.org', 'oecd.org', 'unesco.org',
            'unicef.org', 'undp.org', 'wto.org', 'imf.org', 'europa.eu',
            '.gov', '.edu', '.org'
        ]
        
        for quality_domain in quality_domains:
            if quality_domain in domain:
                score += 0.4
                break
        
        # Academic/research indicators
        academic_indicators = ['journal', 'research', 'study', 'paper', 'academic']
        if any(indicator in text.lower() or indicator in url.lower() for indicator in academic_indicators):
            score += 0.2
        
        # Data/statistics indicators
        data_indicators = ['data', 'statistics', 'report', 'dataset']
        if any(indicator in text.lower() or indicator in url.lower() for indicator in data_indicators):
            score += 0.2
        
        # Recent publication indicators
        if re.search(r'20(2[0-5])', url + text):  # 2020-2025
            score += 0.1
        
        # PDF or document indicators (often more substantial)
        if any(ext in url.lower() for ext in ['.pdf', '.doc', '.report']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _categorize_source(self, url: str, text: str, domain: str) -> str:
        """Categorize the type of source"""
        url_lower = url.lower()
        text_lower = text.lower()
        
        # Government/Official
        if any(indicator in domain for indicator in ['.gov', 'un.org', 'who.int', 'worldbank.org']):
            return "official_government"
        
        # Academic/Research
        if any(indicator in text_lower for indicator in ['journal', 'research', 'study', 'academic']):
            return "academic_research"
        
        # News/Media
        if any(indicator in domain for indicator in ['news', 'times', 'post', 'guardian', 'reuters']):
            return "news_media"
        
        # NGO/Organization
        if '.org' in domain:
            return "organization_ngo"
        
        # Data/Statistics
        if any(indicator in text_lower for indicator in ['data', 'statistics', 'dataset']):
            return "data_statistics"
        
        return "general_web"
    
    def _is_potential_download(self, url: str, text: str) -> bool:
        """Check if citation might be a downloadable resource"""
        download_indicators = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.csv', 'download', 'report', 'dataset', 'attachment'
        ]
        
        combined = f"{url.lower()} {text.lower()}"
        return any(indicator in combined for indicator in download_indicators)
    
    def _assess_citation_sdg_relevance(self, url: str, text: str) -> float:
        """Assess SDG relevance of citation"""
        score = 0.0
        combined = f"{url.lower()} {text.lower()}"
        
        # Direct SDG mentions
        sdg_terms = ['sdg', 'sustainable development', 'agenda 2030', 'goal', 'target']
        score += sum(0.2 for term in sdg_terms if term in combined)
        
        # SDG-related organizations
        sdg_orgs = ['un.org', 'undp.org', 'sustainabledevelopment', 'sdgs.un.org']
        if any(org in url.lower() for org in sdg_orgs):
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_follow_up_questions(self, soup: BeautifulSoup) -> List[str]:
        """Extract follow-up or related questions"""
        questions = []
        
        for selector in self.perplexity_selectors["follow_up_questions"]:
            questions_section = soup.select_one(selector)
            if questions_section:
                # Extract individual questions
                question_elements = questions_section.find_all(['li', 'div', 'p'])
                for elem in question_elements:
                    question_text = elem.get_text(strip=True)
                    if len(question_text) > 10 and question_text.endswith('?'):
                        questions.append(question_text)
        
        return questions[:5]  # Limit to 5 follow-up questions
    
    def _generate_title(self, user_query: str, answer_content: str) -> str:
        """Generate title from query and answer"""
        if user_query and len(user_query) > 5:
            return user_query[:200]
        
        # Fallback to first sentence of answer
        sentences = answer_content.split('.')
        first_sentence = sentences[0].strip() if sentences else answer_content[:100]
        return first_sentence[:150]
    
    def _create_summary(self, content: str) -> str:
        """Create summary from Perplexity answer"""
        # Take first 2-3 sentences
        sentences = re.split(r'[.!?]', content)
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if sentence and char_count + len(sentence) <= 250:
                summary_sentences.append(sentence)
                char_count += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_sentences)
        return summary + '.' if summary and not summary.endswith('.') else summary
    
    def _detect_sdg_relevance(self, content: str) -> List[int]:
        """Detect SDG relevance in content"""
        # Enhanced SDG detection patterns
        sdg_patterns = {
            1: r'\b(poverty|poor|income.inequality|wealth.gap|social.protection|basic.needs)\b',
            2: r'\b(hunger|food.security|malnutrition|agriculture|farming|crop.yield)\b',
            3: r'\b(health|healthcare|medical|disease|mortality|wellbeing|pandemic)\b',
            4: r'\b(education|school|learning|literacy|skills|training|university)\b',
            5: r'\b(gender|women|girls|equality|empowerment|discrimination)\b',
            6: r'\b(water|sanitation|hygiene|drinking.water|clean.water|wastewater)\b',
            7: r'\b(energy|renewable|solar|wind|electricity|clean.energy|fossil)\b',
            8: r'\b(employment|jobs|economic.growth|decent.work|unemployment|labor)\b',
            9: r'\b(infrastructure|innovation|industry|technology|research|development)\b',
            10: r'\b(inequality|inclusion|discrimination|equity|marginalized|income.gap)\b',
            11: r'\b(cities|urban|housing|transport|sustainable.cities|smart.city)\b',
            12: r'\b(consumption|production|waste|recycling|sustainable|circular.economy)\b',
            13: r'\b(climate|carbon|emission|greenhouse|global.warming|adaptation|mitigation)\b',
            14: r'\b(ocean|marine|sea|fisheries|aquatic|coral|plastic.pollution)\b',
            15: r'\b(forest|biodiversity|ecosystem|wildlife|conservation|deforestation)\b',
            16: r'\b(peace|justice|institutions|governance|rule.of.law|corruption)\b',
            17: r'\b(partnership|cooperation|global|development.finance|aid|collaboration)\b'
        }
        
        content_lower = re.sub(r'[^\w\s]', ' ', content.lower())
        relevant_sdgs = []
        
        for sdg_id, pattern in sdg_patterns.items():
            if re.search(pattern, content_lower):
                relevant_sdgs.append(sdg_id)
        
        return relevant_sdgs
    
    def _count_high_quality_sources(self, citations: List[Dict[str, Any]]) -> int:
        """Count high-quality sources in citations"""
        return len([c for c in citations if c.get('quality_score', 0) > 0.6])
    
    def _calculate_perplexity_quality_score(self, content: ExtractedContent,
                                          answer_data: Dict[str, Any],
                                          citations: List[Dict[str, Any]]) -> float:
        """Calculate quality score for Perplexity content"""
        score = 0.0
        
        # Answer quality (0-0.4)
        word_count = answer_data.get('word_count', 0)
        if word_count > 300:
            score += 0.4
        elif word_count > 150:
            score += 0.25
        elif word_count > 75:
            score += 0.1
        
        # Structure quality (0-0.2)
        if answer_data.get('has_lists'):
            score += 0.1
        if answer_data.get('structure_type') in ['structured_list', 'tabular_data']:
            score += 0.1
        
        # Citation quality (0-0.3)
        high_quality_citations = self._count_high_quality_sources(citations)
        if high_quality_citations >= 3:
            score += 0.3
        elif high_quality_citations >= 1:
            score += 0.15
        
        # SDG relevance (0-0.1)
        if len(content.sdg_relevance) >= 2:
            score += 0.1
        elif len(content.sdg_relevance) >= 1:
            score += 0.05
        
        return min(score, 1.0)
