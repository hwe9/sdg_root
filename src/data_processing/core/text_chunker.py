# src/data_processing/core/text_chunker.py
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SDGTextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, model_name: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = SentenceTransformer(model_name)
    
    def smart_chunk_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Intelligently chunks text by sentences while respecting size limits.
        Maintains context and semantic coherence.
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "length": current_length,
                    "embedding": None  # To be filled later
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "length": current_length,
                "embedding": None
            })
            
        return chunks
    
    # def chunk_by_sdg_sections(self, text: str) -> List[Dict[str, Any]]:
    #     """
    #     Chunks text by SDG-specific sections and topics.
    #     Uses your existing SDG keywords for intelligent sectioning.
    #     """
    #     from .keywords import sdg_keywords_dict
        
    #     chunks = []
    #     sections = self._identify_sdg_sections(text, sdg_keywords_dict)
        
    #     for section_name, section_text in sections.items():
    #         if len(section_text) > self.chunk_size:
    #             # Further chunk large sections
    #             sub_chunks = self.smart_chunk_by_sentences(section_text)
    #             for i, sub_chunk in enumerate(sub_chunks):
    #                 sub_chunk.update({
    #                     "sdg_section": section_name,
    #                     "sub_section_id": i
    #                 })
    #                 chunks.append(sub_chunk)
    #         else:
    #             chunks.append({
    #                 "chunk_id": len(chunks),
    #                 "text": section_text,
    #                 "length": len(section_text),
    #                 "sdg_section": section_name,
    #                 "embedding": None
    #             })
        
    #     return chunks

    def chunk_by_sdg_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on SDG-related sections"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_text = ' '.join(words[i:i + self.chunk_size])
            chunks.append({
                "text": chunk_text,
                "chunk_id": i // (self.chunk_size - self.overlap),
                "start_word": i,
                "end_word": min(i + self.chunk_size, len(words))
            })
            
        return chunks
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for each chunk using your existing sentence transformer."""
        for chunk in chunks:
            chunk["embedding"] = self.model.encode(chunk["text"]).tolist()
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get last N characters for overlap."""
        return text[-self.overlap:] if len(text) > self.overlap else text
    
    def _identify_sdg_sections(self, text: str, sdg_keywords: Dict) -> Dict[str, str]:
        """Identify sections of text related to specific SDGs."""
        sections = {"general": ""}
        text_lower = text.lower()
        
        for sdg_name, keywords in sdg_keywords.items():
            section_text = ""
            for keyword in keywords:
                sentences = self._split_into_sentences(text)
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        section_text += sentence + " "
            
            if section_text.strip():
                sections[sdg_name] = section_text.strip()
            else:
                sections["general"] += text 
                
        return sections
