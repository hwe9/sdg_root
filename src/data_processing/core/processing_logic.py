# /sdg_root/src/data_processing/core/processing_logic.py

import re
import os
from typing import Dict
from typing import Any
from typing import List
from .text_chunker import SDGTextChunker
from .keywords import sdg_keywords_dict
from .keywords import ai_keywords_dict
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ProcessingLogic:
    def __init__(self, whisper_model, sentence_model):
        if whisper_model is None:
            logger.warning("Whisper model is None - audio transcription will fail")
        if sentence_model is None:
            raise ValueError("Sentence model cannot be None")
        self.text_chunker = SDGTextChunker(chunk_size=512, overlap=50)
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source='auto', target='en')
        except ImportError:
            logger.warning("GoogleTranslator not available - translation disabled")
            self.translator = None

    
    def process_text_for_ai(self, text_content: str) -> Dict[str, Any]:
        """Process text for AI analysis - single chunk version."""
        embeddings = self.sentence_model.encode(text_content).tolist()
        
        tags = self._extract_tags(text_content)
        
        extracted_info = self.extract_abstract_and_keywords(text_content)
        
        return {
            "embeddings": embeddings,
            "tags": tags,
            "keywords": extracted_info.get('keywords', []),
            "abstract": extracted_info.get('abstract', ''),
            "language": self._detect_language(text_content)
        }

    def detect_language(self, text: str) -> str:
        """Public helper to detect language of given text."""
        return self._detect_language(text)

    def translate_to_english(self, text: str) -> str:
        """Translate text to English if translator is available."""
        if not text or not text.strip():
            return text
        if self.translator is None:
            return text
        try:
            return self.translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed, falling back to original text: {e}")
            return text
    
    def process_text_for_ai_with_chunking(self, text_content: str) -> Dict[str, Any]:
        """Process text with chunking for large documents."""
        if len(text_content) <= 512:
            return self.process_text_for_ai(text_content)
        
        chunks = self.text_chunker.chunk_by_sdg_sections(text_content)
        chunks = self.text_chunker.generate_embeddings_for_chunks(chunks)
        
        processed_chunks = []
        all_tags = set()
        all_embeddings = []
        
        for chunk in chunks:
            chunk_data = self.process_text_for_ai(chunk["text"])
            chunk.update({
                "sdg_tags": chunk_data["tags"],
                "keywords": chunk_data.get("keywords", []),
                "abstract": chunk_data.get("abstract", "")
            })
            processed_chunks.append(chunk)
            all_tags.update(chunk_data["tags"])
            all_embeddings.extend(chunk_data["embeddings"])
        
        if all_embeddings:
            import numpy as np
            combined_embeddings = np.mean(
                np.array(all_embeddings).reshape(len(processed_chunks), -1), 
                axis=0
            ).tolist()
        else:
            combined_embeddings = []
        
        return {
            "chunks": processed_chunks,
            "combined_tags": list(all_tags),
            "combined_embeddings": combined_embeddings,
            "total_chunks": len(processed_chunks),
            "total_length": len(text_content)
        }
    
    def transcribe_audio(self, audio_path: str) -> str:
    
    
        # Model-Validierung
        if not self.whisper_model:
            raise ValueError("Whisper model not initialized")
        
        # Datei-Validierung
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Dateigröße-Validierung (max 50MB)
        file_size = os.path.getsize(audio_path)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise ValueError(f"Audio file too large: {file_size / 1024 / 1024:.1f}MB > 50MB")
        
        try:
            # Timeout für Transkription (max 5 Minuten)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Transcription timeout after 5 minutes")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 Minuten
            
            segments, info = self.whisper_model.transcribe(audio_path)
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            signal.alarm(0)  # Timeout zurücksetzen
            
            result = transcription.strip()
            if not result:
                logger.warning(f"Empty transcription for {audio_path}")
            
            return result
            
        except TimeoutError:
            logger.error(f"Transcription timeout for {audio_path}")
            return ""
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return ""
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract SDG and AI tags from text."""
        text_lower = text.lower()
        tags = []
        
        # Extract SDG tags
        for sdg_name, keywords in sdg_keywords_dict.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    tags.append(sdg_name)
                    break
        
        # Extract AI tags
        for ai_name, keywords in ai_keywords_dict.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    tags.append(ai_name)
                    break
        
        return list(set(tags))
    
    def extract_abstract_and_keywords(self, text: str) -> Dict[str, Any]:
        """Extract abstract and keywords from text."""
        paragraphs = text.split('\n\n')
        abstract = paragraphs[0][:300] + "..." if len(paragraphs[0]) > 300 else paragraphs[0]

        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in ['that', 'with', 'have', 'this', 'will', 'from', 'they', 'been', 'their']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return {
            "abstract": abstract,
            "keywords": keywords
        }
    
    def _detect_language(self, text: str) -> str:
        """Enhanced language detection including Chinese and Hindi."""
        sample = text[:500].lower()  # Increased sample size for better detection
        
        # English indicators
        if any(word in sample for word in ['the', 'and', 'that', 'with', 'have', 'this', 'will', 'from']):
            return 'en'
        
        # German indicators
        elif any(word in sample for word in ['der', 'die', 'das', 'und', 'mit', 'eine', 'einer', 'auch']):
            return 'de'
        
        # French indicators
        elif any(word in sample for word in ['le', 'la', 'et', 'des', 'les', 'une', 'dans', 'pour']):
            return 'fr'
        
        # Spanish indicators
        elif any(word in sample for word in ['el', 'la', 'los', 'las', 'que', 'con', 'una', 'para']):
            return 'es'
        
        # Chinese indicators - check for Chinese characters
        elif any('\u4e00' <= char <= '\u9fff' for char in text[:200]):  # CJK Unified Ideographs
            return 'zh'
        
        # Hindi indicators - check for Devanagari script
        elif any('\u0900' <= char <= '\u097f' for char in text[:200]):  # Devanagari Unicode block
            return 'hi'
        
        # Additional Chinese detection (Traditional Chinese)
        elif any('\u3400' <= char <= '\u4dbf' for char in text[:200]):  # CJK Extension A
            return 'zh'
        
        # Hindi common words in Latin script (transliterated)
        elif any(word in sample for word in ['hai', 'hain', 'mein', 'aur', 'kya', 'koi', 'yeh', 'woh']):
            return 'hi'
        
        # Chinese common words in Pinyin (romanized)
        elif any(word in sample for word in ['shi', 'zai', 'you', 'wei', 'dui', 'gen', 'cong']):
            return 'zh'
        
        # Default to English if no clear detection
        else:
            return 'en'

    def detect_language_advanced(self, text: str) -> Dict[str, Any]:
        """
        Advanced language detection with confidence scores
        """
        sample = text[:1000]  
        
        language_patterns = {
            'en': {
                'common_words': ['the', 'and', 'that', 'with', 'have', 'this', 'will', 'from', 'they', 'been'],
                'weight': 0
            },
            'de': {
                'common_words': ['der', 'die', 'das', 'und', 'mit', 'eine', 'einer', 'auch', 'sich', 'aber'],
                'weight': 0
            },
            'fr': {
                'common_words': ['le', 'la', 'et', 'des', 'les', 'une', 'dans', 'pour', 'qui', 'avec'],
                'weight': 0
            },
            'es': {
                'common_words': ['el', 'la', 'los', 'las', 'que', 'con', 'una', 'para', 'por', 'como'],
                'weight': 0
            },
            'zh': {
                'common_words': [],  # Will use character detection
                'weight': 0
            },
            'hi': {
                'common_words': ['hai', 'hain', 'mein', 'aur', 'kya', 'koi', 'yeh', 'woh', 'iska', 'jab'],
                'weight': 0
            }
        }
        
        sample_lower = sample.lower()
        
        # Count word matches for Latin-script languages
        for lang_code, lang_data in language_patterns.items():
            if lang_code in ['zh', 'hi']:  # Skip for now, handle separately
                continue
            
            word_matches = sum(1 for word in lang_data['common_words'] if word in sample_lower)
            lang_data['weight'] = word_matches
        
        # Chinese character detection
        chinese_chars = sum(1 for char in sample if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf')
        if chinese_chars > 10:  # Threshold for Chinese detection
            language_patterns['zh']['weight'] = chinese_chars * 2  # Give higher weight
        
        # Hindi Devanagari script detection
        hindi_chars = sum(1 for char in sample if '\u0900' <= char <= '\u097f')
        if hindi_chars > 5:  # Threshold for Hindi detection
            language_patterns['hi']['weight'] = hindi_chars * 3  # Give higher weight
        else:
            # Check romanized Hindi words
            hindi_word_matches = sum(1 for word in language_patterns['hi']['common_words'] if word in sample_lower)
            language_patterns['hi']['weight'] = hindi_word_matches
        
        # Find language with highest weight
        detected_lang = max(language_patterns.keys(), key=lambda x: language_patterns[x]['weight'])
        confidence = language_patterns[detected_lang]['weight']
        
        # Normalize confidence score
        max_possible_score = len(sample.split()) * 0.1  # Rough estimate
        normalized_confidence = min(confidence / max(max_possible_score, 1), 1.0)
        
        return {
            'language': detected_lang,
            'confidence': normalized_confidence,
            'scores': {lang: data['weight'] for lang, data in language_patterns.items()},
            'method': 'advanced_pattern_matching'
        }
    
    def extract_abstract_and_keywords(self, text: str) -> Dict[str, Any]:
        """Extract abstract and keywords from text content"""
        
        # Detect language first
        language_info = self.detect_language_advanced(text)
        
        # Extract potential abstract (first few sentences)
        sentences = re.split(r'[.!?]', text)
        abstract_candidates = []
        
        for i, sentence in enumerate(sentences[:5]):  
            sentence = sentence.strip()
            if len(sentence) > 50 and len(sentence) < 500:  
                abstract_candidates.append(sentence)
        
        abstract = '. '.join(abstract_candidates[:3]) if abstract_candidates else text[:300]
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        
        stop_words = {
            'en': ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'each', 'which'],
            'de': ['dass', 'sich', 'aber', 'auch', 'noch', 'nach', 'beim', 'dann', 'kann', 'wird'],
            'fr': ['dans', 'pour', 'avec', 'sont', 'plus', 'tout', 'cette', 'peut', 'comme', 'fait'],
            'es': ['para', 'como', 'este', 'esta', 'pero', 'todo', 'hace', 'muy', 'ahora', 'cada'],
            'zh': [],  # Chinese doesn't use space-separated words in the same way
            'hi': ['hain', 'kiya', 'jata', 'karne', 'hota', 'raha', 'gaya', 'kuch', 'baat', 'saat']
        }
        
        current_stop_words = stop_words.get(language_info['language'], stop_words['en'])
        
        for word in words:
            if word not in current_stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return {
            'abstract': abstract,
            'keywords': keywords,
            'language_info': language_info,
            'word_count': len(text.split()),
            'character_count': len(text)
        }

 
