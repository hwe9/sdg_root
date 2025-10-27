"""
Gemini 2.5 content analysis extractor
Processes content through Gemini for SDG analysis
"""
import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
import asyncio
import json
import re
from .base_extractor import BaseExtractor
from .base_extractor import ExtractedContent

logger = logging.getLogger(__name__)

class GeminiExtractor(BaseExtractor):
    """
    Extractor for Gemini 2.5 analysis results
    Processes existing content through Gemini for enhanced SDG insights
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gemini_prompts = self._load_gemini_prompts()
        self.sdg_keywords = self._load_sdg_keywords()
    
    def _load_gemini_prompts(self) -> Dict[str, str]:
        """Load Gemini analysis prompts"""
        return {
            "sdg_analysis": """
            Analyze the following text for relevance to the UN Sustainable Development Goals (SDGs).
            
            Text: {content}
            
            Please provide:
            1. Primary SDG goals (1-17) that this content addresses
            2. Confidence score (0-1) for each identified SDG
            3. Key themes and topics
            4. Regional relevance if mentioned
            5. Summary of SDG-related content
            
            Format your response as JSON with the following structure:
            {{
                "sdg_goals": [{{goal_number, confidence_score}}],
                "themes": ["theme1", "theme2"],
                "region": "region_name",
                "summary": "content_summary",
                "quality_indicators": {{
                    "has_data": boolean,
                    "has_citations": boolean,
                    "policy_relevant": boolean
                }}
            }}
            """,
            
            "content_enhancement": """
            Enhance and summarize the following content for SDG research purposes:
            
            Original Content: {content}
            
            Please provide:
            1. A concise summary (200-300 words)
            2. Key findings or insights
            3. Methodology mentioned (if any)
            4. Data sources referenced
            5. Policy implications
            
            Focus on aspects relevant to sustainable development research.
            """,
            
            "quality_assessment": """
            Assess the quality and credibility of this content:
            
            Content: {content}
            Source: {source_url}
            
            Evaluate:
            1. Scientific rigor (0-10)
            2. Data reliability (0-10) 
            3. Bias indicators
            4. Citation quality
            5. Overall credibility score (0-10)
            
            Provide reasoning for each score.
            """
        }
    
    def _load_sdg_keywords(self) -> Dict[int, List[str]]:
        """Load SDG-specific keywords for validation"""
        # This would typically load from your keywords.py file
        return {
            1: ["poverty", "income inequality", "social protection"],
            2: ["hunger", "food security", "nutrition", "agriculture"],
            3: ["health", "mortality", "disease", "healthcare"],
            4: ["education", "learning", "literacy", "skills"],
            5: ["gender", "women", "girls", "equality"],
            6: ["water", "sanitation", "hygiene"],
            7: ["energy", "renewable", "electricity", "clean"],
            8: ["employment", "economic growth", "decent work"],
            9: ["infrastructure", "innovation", "industry"],
            10: ["inequality", "inclusion", "discrimination"],
            11: ["cities", "urban", "housing", "transport"],
            12: ["consumption", "production", "waste", "sustainability"],
            13: ["climate", "greenhouse gas", "adaptation"],
            14: ["ocean", "marine", "fisheries"],
            15: ["forest", "biodiversity", "ecosystem"],
            16: ["peace", "justice", "institutions", "governance"],
            17: ["partnership", "cooperation", "finance"]
        }
    
    def validate_source(self, source_url: str) -> bool:
        """Validate if source is suitable for Gemini analysis"""
        # Accept any content for Gemini analysis
        return True
    
    async def extract(self, content_input: str, **kwargs) -> List[ExtractedContent]:
        """
        Process content through Gemini analysis
        content_input can be raw text, URL, or structured content
        """
        try:
            # Determine input type
            if content_input.startswith(('http://', 'https://')):
                # URL input - fetch content first
                content_text = await self._fetch_url_content(content_input)
                source_url = content_input
            else:
                # Direct text input
                content_text = content_input
                source_url = kwargs.get('source_url', '')
            
            if not content_text or len(content_text.strip()) < 100:
                logger.warning("Content too short for Gemini analysis")
                return []
            
            # Process through Gemini (simulated - replace with actual Gemini API)
            analysis_result = await self._simulate_gemini_analysis(content_text)
            
            # Create enhanced content object
            enhanced_content = ExtractedContent(
                title=self._extract_title(content_text),
                content=content_text,
                summary=analysis_result.get('summary', ''),
                url=source_url,
                source_type="gemini_analysis",
                language=kwargs.get('language', 'en'),
                region=analysis_result.get('region', ''),
                metadata={
                    'gemini_analysis': analysis_result,
                    'processing_method': 'gemini_2.5',
                    'themes': analysis_result.get('themes', []),
                    'quality_indicators': analysis_result.get('quality_indicators', {})
                },
                sdg_relevance=[goal['goal_number'] for goal in analysis_result.get('sdg_goals', [])]
            )
            
            # Calculate quality score
            enhanced_content.quality_score = self._calculate_gemini_quality_score(analysis_result)
            
            return [enhanced_content]
            
        except Exception as e:
            logger.error(f"Error in Gemini extraction: {e}")
            return []
    
    async def _fetch_url_content(self, url: str) -> str:
        """Fetch content from URL for analysis"""
        response = await self.fetch_with_retry(url)
        if response:
            content = await response.text()
            # Basic HTML cleaning
            clean_content = re.sub(r'<[^>]+>', ' ', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            return clean_content
        return ""
    
    async def _simulate_gemini_analysis(self, content: str) -> Dict[str, Any]:
        """
        Simulate Gemini 2.5 analysis
        Replace this with actual Gemini API calls
        """
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Simple SDG detection based on keywords
        detected_sdgs = []
        for sdg_id, keywords in self.sdg_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
            if keyword_matches > 0:
                confidence = min(keyword_matches / len(keywords), 1.0)
                if confidence > 0.1:  # Minimum confidence threshold
                    detected_sdgs.append({
                        "goal_number": sdg_id,
                        "confidence_score": confidence
                    })
        
        # Generate summary (first 300 chars as simulation)
        summary = content[:300] + "..." if len(content) > 300 else content
        
        # Extract themes (simple word frequency analysis)
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        themes = [theme[0] for theme in themes]
        
        # Quality indicators
        quality_indicators = {
            "has_data": any(word in content.lower() for word in ['data', 'statistics', 'research', 'study']),
            "has_citations": any(word in content.lower() for word in ['doi', 'reference', 'citation', 'source']),
            "policy_relevant": any(word in content.lower() for word in ['policy', 'government', 'legislation', 'regulation'])
        }
        
        return {
            "sdg_goals": detected_sdgs,
            "themes": themes,
            "region": self._detect_region(content),
            "summary": summary,
            "quality_indicators": quality_indicators
        }
    
    def _detect_region(self, content: str) -> str:
        """Simple region detection"""
        regions = {
            "EU": ["europe", "european", "eu", "brussels"],
            "USA": ["united states", "america", "usa", "us"],
            "China": ["china", "chinese", "beijing"],
            "India": ["india", "indian", "delhi"],
            "ASEAN": ["asean", "southeast asia", "vietnam", "thailand"],
            "BRICS": ["brics", "brazil", "russia", "south africa"]
        }
        
        content_lower = content.lower()
        for region, keywords in regions.items():
            if any(keyword in content_lower for keyword in keywords):
                return region
        
        return ""
    
    def _extract_title(self, content: str) -> str:
        """Extract or generate title from content"""
        # Simple title extraction - first sentence or first 100 chars
        sentences = content.split('.')
        if sentences and len(sentences[0].strip()) > 10:
            return sentences[0].strip()[:100]
        return content[:100].strip()
    
    def _calculate_gemini_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate quality score based on Gemini analysis"""
        score = 0.0
        
        # SDG relevance (0-0.4)
        sdg_goals = analysis.get('sdg_goals', [])
        if sdg_goals:
            avg_confidence = sum(goal['confidence_score'] for goal in sdg_goals) / len(sdg_goals)
            score += avg_confidence * 0.4
        
        # Quality indicators (0-0.3)
        quality_indicators = analysis.get('quality_indicators', {})
        quality_count = sum(quality_indicators.values())
        score += (quality_count / 3) * 0.3
        
        # Themes richness (0-0.2)
        themes = analysis.get('themes', [])
        if len(themes) >= 3:
            score += 0.2
        elif len(themes) >= 1:
            score += 0.1
        
        # Regional relevance (0-0.1)
        if analysis.get('region'):
            score += 0.1
        
        return min(score, 1.0)
    
    async def analyze_existing_content(self, content_items: List[Dict[str, Any]]) -> List[ExtractedContent]:
        """Analyze existing content items through Gemini"""
        results = []
        
        for item in content_items:
            content_text = item.get('content', '')
            if content_text:
                enhanced = await self.extract(
                    content_text,
                    source_url=item.get('url', ''),
                    language=item.get('language', 'en')
                )
                results.extend(enhanced)
        
        return results
