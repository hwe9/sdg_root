"""
Intelligent SDG-aware text chunking with semantic boundaries
"""
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SDGSemanticChunker:
    """Advanced chunking with SDG context preservation"""
    
    def __init__(self, 
                 target_chunk_size: int = 512,
                 overlap_size: int = 50,
                 min_chunk_size: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size  
        self.min_chunk_size = min_chunk_size
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # SDG-spezifische Breakpoints
        self.sdg_markers = [
            r"SDG\s*\d+", r"Goal\s*\d+", r"Target\s*\d+\.\d+",
            r"sustainable development", r"agenda 2030"
        ]
    
    def smart_chunk(self, text: str, preserve_sdg_context: bool = True) -> List[Dict[str, Any]]:
        """Semantically-aware chunking with SDG context preservation"""
        
        # 1. Identify SDG sections
        sdg_sections = self._identify_sdg_sections(text) if preserve_sdg_context else []
        
        # 2. Split into sentences
        sentences = self._split_sentences(text)
        
        # 3. Create semantic chunks
        chunks = self._create_semantic_chunks(sentences, sdg_sections)
        
        # 4. Post-process chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "chunk_id": i,
                "text": chunk["text"],
                "start_position": chunk["start_pos"],
                "end_position": chunk["end_pos"], 
                "sdg_context": chunk.get("sdg_context", []),
                "semantic_coherence": chunk.get("coherence_score", 0.0),
                "word_count": len(chunk["text"].split()),
                "sentence_count": len(chunk.get("sentences", [])),
                "embedding": None  # Wird später gefüllt
            })
        
        return processed_chunks
    
    def _identify_sdg_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify SDG-specific sections in text"""
        sections = []
        
        for pattern in self.sdg_markers:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = match.start()
                # Erweitere Kontext um den SDG-Marker
                context_start = max(0, start - 200)
                context_end = min(len(text), start + 500)
                
                sections.append({
                    "start": context_start,
                    "end": context_end,
                    "marker": match.group(),
                    "context": text[context_start:context_end]
                })
        
        return sections
    
    def _create_semantic_chunks(self, sentences: List[str], sdg_sections: List[Dict]) -> List[Dict]:
        """Create chunks with semantic boundary detection"""
        chunks = []
        current_chunk = []
        current_length = 0
        sentence_embeddings = None
        
        # Generate sentence embeddings für Semantic Similarity
        if len(sentences) > 1:
            sentence_embeddings = self.sentence_model.encode(sentences)
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence.split())
            
            # Check if adding sentence exceeds target size
            if current_length + sentence_length > self.target_chunk_size and current_chunk:
                
                # Find optimal split point using semantic similarity
                split_point = self._find_optimal_split(
                    current_chunk, sentence_embeddings[i-len(current_chunk):i] if sentence_embeddings is not None else None
                )
                
                # Create chunk
                chunk_text = ' '.join(current_chunk[:split_point])
                chunks.append({
                    "text": chunk_text,
                    "sentences": current_chunk[:split_point],
                    "start_pos": i - len(current_chunk),
                    "end_pos": i - len(current_chunk) + split_point,
                    "coherence_score": self._calculate_coherence(current_chunk[:split_point])
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[max(0, split_point - self.overlap_size // 20):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
            
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": ' '.join(current_chunk),
                "sentences": current_chunk,
                "start_pos": len(sentences) - len(current_chunk),
                "end_pos": len(sentences),
                "coherence_score": self._calculate_coherence(current_chunk)
            })
        
        return chunks
    
    def _find_optimal_split(self, sentences: List[str], embeddings: np.ndarray = None) -> int:
        """Find optimal split point using semantic similarity"""
        if len(sentences) <= 2 or embeddings is None:
            return len(sentences) // 2
        
        # Calculate semantic discontinuity
        similarities = []
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            similarities.append(similarity)
        
        # Find point with lowest similarity (highest discontinuity)
        if similarities:
            min_similarity_idx = np.argmin(similarities)
            return min_similarity_idx + 1
        
        return len(sentences) // 2
    
    def _calculate_coherence(self, sentences: List[str]) -> float:
        """Calculate semantic coherence of chunk"""
        if len(sentences) <= 1:
            return 1.0
        
        try:
            embeddings = self.sentence_model.encode(sentences)
            similarities = []
            
            for i in range(1, len(embeddings)):
                sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.5
        except:
            return 0.5
    
    def _split_sentences(self, text: str) -> List[str]:
        """Advanced sentence splitting with SDG-context awareness"""
        # Verbesserte Sentence Segmentation
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]
