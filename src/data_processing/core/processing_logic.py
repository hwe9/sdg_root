# Enhanced processing_logic.py
import re
from .text_chunker import SDGTextChunker
from .keywords import sdg_keywords_dict
from deep_translator import GoogleTranslator
import pytesseract
from PIL import Image

class ProcessingLogic:
    def __init__(self, whisper_model, sentence_model):
        self.whisper_model = whisper_model
        self.sentence_model = sentence_model
        self.text_chunker = SDGTextChunker(chunk_size=512, overlap=50)
    
    def process_text_for_ai_with_chunking(self, text_content: str) -> Dict[str, Any]:
        
        if len(text_content) <= 512:
            return self._process_single_chunk(text_content)
        
        chunks = self.text_chunker.chunk_by_sdg_sections(text_content)
        chunks = self.text_chunker.generate_embeddings_for_chunks(chunks)
        
        processed_chunks = []
        all_tags = set()
        
        for chunk in chunks:
            chunk_data = self._process_single_chunk(chunk["text"])
            chunk.update({
                "sdg_tags": chunk_data["tags"],
                "keywords": chunk_data.get("keywords", []),
                "abstract": chunk_data.get("abstract")
            })
            processed_chunks.append(chunk)
            all_tags.update(chunk_data["tags"])
        
        return {
            "chunks": processed_chunks,
            "combined_tags": list(all_tags),
            "total_chunks": len(processed_chunks),
            "total_length": len(text_content)
        }
    
    def _process_single_chunk(self, text: str) -> Dict[str, Any]:
        """Process a single chunk (your existing logic)."""
        embeddings = self.sentence_model.encode(text).tolist()
        tags = []
        
        text_lower = text.lower()
        for sdg_name, keywords in sdg_keywords_dict.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    tags.append(sdg_name)
                    break
        
        extracted_info = self.extract_abstract_and_keywords(text)
        tags.extend(extracted_info.get('keywords', []))
        
        return {
            "text": text,
            "embeddings": embeddings,
            "tags": list(set(tags)),
            "abstract": extracted_info.get('abstract'),
            "keywords": extracted_info.get('keywords', [])
        }
