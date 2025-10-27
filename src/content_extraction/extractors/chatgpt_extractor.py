"""
ChatGPT HTML Result Extractor (Non-API)
Parses ChatGPT web interface results and extracts content with follow-up links
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

class ChatGPTExtractor(BaseExtractor):
    """
    ChatGPT web interface result extractor
    Parses ChatGPT conversation HTML and extracts insights with source links
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chatgpt_selectors = self._load_chatgpt_selectors()
        self.link_keywords = self._load_link_keywords()
    
    def _load_chatgpt_selectors(self) -> Dict[str, List[str]]:
        """Load ChatGPT-specific HTML selectors"""
        return {
            "message_containers": [
                "div[data-message-author-role='assistant']",
                ".message-content",
                ".markdown",
                "[data-testid='conversation-turn-message']",
                ".prose"
            ],
            "user_messages": [
                "div[data-message-author-role='user']",
                ".user-message",
                "[data-testid='user-message']"
            ],
            "assistant_messages": [
                "div[data-message-author-role='assistant']", 
                ".assistant-message",
                "[data-testid='assistant-message']"
            ],
            "code_blocks": [
                "pre code",
                ".code-block", 
                "pre"
            ],
            "citations": [
                ".citation",
                "sup a",
                "[data-citation]"
            ]
        }
    
    def _load_link_keywords(self) -> Dict[str, List[str]]:
        """Keywords to identify valuable links"""
        return {
            "download_links": [
                "download", "pdf", "report", "study", "data", "dataset", 
                "excel", "csv", "doc", "presentation", "whitepaper"
            ],
            "reference_links": [
                "source", "reference", "cite", "study", "research", 
                "article", "paper", "journal", "publication"
            ],
            "official_links": [
                "gov", "un.org", "who.int", "worldbank.org", "oecd.org",
                "europa.eu", "unicef.org", "undp.org", "unesco.org"
            ],
            "sdg_links": [
                "sdg", "sustainable", "development", "goal", "target",
                "indicator", "progress", "agenda"
            ]
        }
    
    def validate_source(self, source_url: str) -> bool:
        """Validate if URL is from ChatGPT interface"""
        chatgpt_domains = [
            'chat.openai.com',
            'chatgpt.com',
            'openai.com/chat'
        ]
        
        parsed = urlparse(source_url.lower())
        return any(domain in parsed.netloc for domain in chatgpt_domains)
    
    async def extract(self, source_url: str, **kwargs) -> List[ExtractedContent]:
        """Extract content from ChatGPT conversation page"""
        try:
            response = await self.fetch_with_retry(source_url)
            if not response:
                return []
            
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            self._clean_soup(soup)
            
            # Extract conversation turns
            conversation_data = self._extract_conversation(soup, source_url)
            
            # Extract follow-up links and resources
            extracted_links = self._extract_valuable_links(soup, source_url)
            
            # Create structured content
            extracted_items = []
            
            for turn_data in conversation_data:
                if turn_data['role'] == 'assistant' and len(turn_data['content']) > 100:
                    extracted_content = ExtractedContent(
                        title=self._generate_title(turn_data['content']),
                        content=turn_data['content'],
                        summary=self._create_summary(turn_data['content']),
                        url=source_url,
                        source_type="chatgpt_result",
                        language=kwargs.get('language', 'en'),
                        region=kwargs.get('region', ''),
                        metadata={
                            'conversation_turn': turn_data['turn_number'],
                            'extracted_links': extracted_links,
                            'user_query': turn_data.get('user_query', ''),
                            'extraction_method': 'chatgpt_html_parser',
                            'contains_code': turn_data.get('has_code', False),
                            'contains_citations': turn_data.get('has_citations', False),
                            'response_length': len(turn_data['content'])
                        }
                    )
                    
                    # Detect SDG relevance
                    extracted_content.sdg_relevance = self._detect_sdg_relevance(turn_data['content'])
                    
                    # Calculate quality score
                    extracted_content.quality_score = self._calculate_chatgpt_quality_score(
                        extracted_content, turn_data, extracted_links
                    )
                    
                    extracted_items.append(extracted_content)
            
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error extracting ChatGPT content from {source_url}: {e}")
            return []
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Remove unwanted elements from soup"""
        # Remove scripts, styles, and navigation
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _extract_conversation(self, soup: BeautifulSoup, source_url: str) -> List[Dict[str, Any]]:
        """Extract conversation turns from ChatGPT interface"""
        conversation_turns = []
        
        # Try different selector strategies
        messages = []
        for selector_group in self.chatgpt_selectors["message_containers"]:
            found_messages = soup.select(selector_group)
            if found_messages:
                messages = found_messages
                break
        
        if not messages:
            # Fallback: look for common conversation patterns
            messages = soup.find_all(['div', 'article'], class_=re.compile(r'(message|conversation|chat|response)'))
        
        current_user_query = ""
        turn_number = 1
        
        for i, message in enumerate(messages):
            # Determine if this is user or assistant message
            message_text = message.get_text(separator=' ', strip=True)
            if len(message_text) < 20:  # Skip very short messages
                continue
            
            # Check message role
            role = self._determine_message_role(message, message_text)
            
            if role == 'user':
                current_user_query = message_text
            elif role == 'assistant':
                # Check for code blocks
                has_code = bool(message.find_all(['pre', 'code']))
                
                # Check for citations or links
                has_citations = bool(message.find_all('a') or message.find_all(['sup', 'cite']))
                
                conversation_turns.append({
                    'turn_number': turn_number,
                    'role': 'assistant',
                    'content': message_text,
                    'user_query': current_user_query,
                    'has_code': has_code,
                    'has_citations': has_citations,
                    'html_element': message
                })
                turn_number += 1
        
        return conversation_turns
    
    def _determine_message_role(self, message_element, message_text: str) -> str:
        """Determine if message is from user or assistant"""
        # Check data attributes first
        role_attr = message_element.get('data-message-author-role')
        if role_attr:
            return role_attr
        
        # Check class names
        class_names = ' '.join(message_element.get('class', [])).lower()
        if any(term in class_names for term in ['user', 'human']):
            return 'user'
        elif any(term in class_names for term in ['assistant', 'ai', 'bot']):
            return 'assistant'
        
        # Heuristic based on content patterns
        if re.match(r'^(what|how|why|can you|please|could you)', message_text.lower()):
            return 'user'
        elif len(message_text) > 200:  # Assistant responses tend to be longer
            return 'assistant'
        
        return 'unknown'
    
    def _extract_valuable_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract and categorize valuable links from the page"""
        links = []
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True)
            
            # Skip empty links or javascript
            if not href or href.startswith(('javascript:', '#', 'mailto:')):
                continue
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = urljoin(base_url, href)
            elif not href.startswith(('http://', 'https://')):
                continue
            
            # Categorize link
            link_category = self._categorize_link(href, link_text)
            
            if link_category:  # Only include categorized links
                links.append({
                    'url': href,
                    'text': link_text,
                    'category': link_category,
                    'domain': urlparse(href).netloc,
                    'is_download': self._is_download_link(href, link_text),
                    'sdg_relevance': self._assess_link_sdg_relevance(href, link_text)
                })
        
        # Sort by relevance (SDG relevance + category priority)
        links.sort(key=lambda x: (x['sdg_relevance'], x['is_download']), reverse=True)
        
        return links[:20]  # Limit to most relevant links
    
    def _categorize_link(self, url: str, link_text: str) -> Optional[str]:
        """Categorize link based on URL and text content"""
        url_lower = url.lower()
        text_lower = link_text.lower()
        combined_text = f"{url_lower} {text_lower}"
        
        # Check for official/government sources
        if any(domain in url_lower for domain in self.link_keywords["official_links"]):
            return "official_source"
        
        # Check for download links
        if any(keyword in combined_text for keyword in self.link_keywords["download_links"]):
            return "download_resource"
        
        # Check for reference/research links
        if any(keyword in combined_text for keyword in self.link_keywords["reference_links"]):
            return "research_reference"
        
        # Check for SDG-specific links
        if any(keyword in combined_text for keyword in self.link_keywords["sdg_links"]):
            return "sdg_resource"
        
        # Check file extensions
        if any(ext in url_lower for ext in ['.pdf', '.doc', '.xls', '.ppt', '.csv']):
            return "document_file"
        
        return None
    
    def _is_download_link(self, url: str, link_text: str) -> bool:
        """Check if link is likely a download"""
        download_indicators = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.csv', '.zip', '.tar', '.gz', 'download', 'attachment'
        ]
        
        combined = f"{url.lower()} {link_text.lower()}"
        return any(indicator in combined for indicator in download_indicators)
    
    def _assess_link_sdg_relevance(self, url: str, link_text: str) -> float:
        """Assess how relevant a link is to SDG content"""
        score = 0.0
        combined_text = f"{url.lower()} {link_text.lower()}"
        
        # Official SDG sources get high score
        sdg_domains = ['un.org', 'sdgs.un.org', 'sustainabledevelopment.un.org']
        if any(domain in url.lower() for domain in sdg_domains):
            score += 0.5
        
        # SDG keywords
        sdg_terms = ['sdg', 'sustainable development', 'agenda 2030', 'goal', 'target']
        score += sum(0.1 for term in sdg_terms if term in combined_text)
        
        # Research/data indicators
        research_terms = ['research', 'data', 'report', 'study', 'analysis']
        score += sum(0.05 for term in research_terms if term in combined_text)
        
        return min(score, 1.0)
    
    def _generate_title(self, content: str) -> str:
        """Generate title from ChatGPT response content"""
        # Look for clear topic indicators
        sentences = content.split('.')
        first_sentence = sentences[0].strip() if sentences else content[:100]
        
        # Remove common ChatGPT prefixes
        prefixes_to_remove = [
            "Based on", "According to", "Here's", "Here are", "I'll help you",
            "Let me", "To answer", "In response"
        ]
        
        for prefix in prefixes_to_remove:
            if first_sentence.startswith(prefix):
                # Try to find the actual topic
                remaining = first_sentence[len(prefix):].strip()
                if len(remaining) > 20:
                    first_sentence = remaining
                break
        
        # Clean up and limit length
        title = re.sub(r'^[^a-zA-Z0-9]*', '', first_sentence)
        return title[:150] if len(title) > 150 else title
    
    def _create_summary(self, content: str) -> str:
        """Create summary from ChatGPT response"""
        # Take first 2-3 sentences or up to 300 characters
        sentences = re.split(r'[.!?]', content)
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if sentence and char_count + len(sentence) <= 300:
                summary_sentences.append(sentence)
                char_count += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_sentences)
        return summary + '.' if summary and not summary.endswith('.') else summary
    
    def _detect_sdg_relevance(self, content: str) -> List[int]:
        """Detect SDG relevance in ChatGPT response"""
        sdg_patterns = {
            1: r'\b(poverty|poor|income|wealth|social.protection)\b',
            2: r'\b(hunger|food|nutrition|agriculture|farming)\b',
            3: r'\b(health|healthcare|medical|disease|mortality)\b',
            4: r'\b(education|learning|school|literacy|skills)\b',
            5: r'\b(gender|women|girls|equality|empowerment)\b',
            6: r'\b(water|sanitation|hygiene|clean.water)\b',
            7: r'\b(energy|renewable|electricity|clean.energy)\b',
            8: r'\b(employment|jobs|economic.growth|decent.work)\b',
            9: r'\b(infrastructure|innovation|industry|technology)\b',
            10: r'\b(inequality|inclusion|discrimination|equity)\b',
            11: r'\b(cities|urban|housing|transport|sustainable.cities)\b',
            12: r'\b(consumption|production|waste|recycling|sustainable)\b',
            13: r'\b(climate|carbon|emission|greenhouse|warming)\b',
            14: r'\b(ocean|marine|sea|fisheries|aquatic)\b',
            15: r'\b(forest|biodiversity|ecosystem|wildlife|conservation)\b',
            16: r'\b(peace|justice|institutions|governance|law)\b',
            17: r'\b(partnership|cooperation|global|development.finance)\b'
        }
        
        content_lower = re.sub(r'[^\w\s]', ' ', content.lower())
        relevant_sdgs = []
        
        for sdg_id, pattern in sdg_patterns.items():
            if re.search(pattern, content_lower):
                relevant_sdgs.append(sdg_id)
        
        return relevant_sdgs
    
    def _calculate_chatgpt_quality_score(self, content: ExtractedContent, 
                                       turn_data: Dict[str, Any],
                                       extracted_links: List[Dict[str, Any]]) -> float:
        """Calculate quality score for ChatGPT extracted content"""
        score = 0.0
        
        # Base content quality (0-0.4)
        content_length = len(content.content)
        if content_length > 500:
            score += 0.4
        elif content_length > 200:
            score += 0.2
        
        # Has structured elements (0-0.2)
        if turn_data.get('has_code'):
            score += 0.1
        if turn_data.get('has_citations'):
            score += 0.1
        
        # Link quality (0-0.2)
        if extracted_links:
            high_quality_links = [l for l in extracted_links if l.get('sdg_relevance', 0) > 0.3]
            if len(high_quality_links) >= 3:
                score += 0.2
            elif len(high_quality_links) >= 1:
                score += 0.1
        
        # SDG relevance (0-0.1)
        if len(content.sdg_relevance) >= 2:
            score += 0.1
        elif len(content.sdg_relevance) >= 1:
            score += 0.05
        
        # User query context (0-0.1)
        user_query = turn_data.get('user_query', '')
        if len(user_query) > 20:  # Had meaningful user question
            score += 0.1
        
        return min(score, 1.0)
