"""
Advanced embedding models for multilingual SDG content
Extracted and enhanced from your text_vektorizer.py and processing_logic.py
"""
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings for input texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        pass

class SentenceTransformerModel(BaseEmbeddingModel):
    """Multilingual Sentence Transformer for SDG content"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded SentenceTransformer model: {model_name}")
    
    def encode(self, texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: int = 32,
               **kwargs) -> np.ndarray:
        """Generate embeddings with batch processing"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            **kwargs
        )
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embeddings for high-quality SDG analysis"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        self.model_name = model_name
        self.dimension = 1536 if model_name == "text-embedding-ada-002" else 1536
        if api_key:
            openai.api_key = api_key
        logger.info(f"Initialized OpenAI embedding model: {model_name}")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate OpenAI embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            embeddings = np.array([item['embedding'] for item in response['data']])
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension

class SDGSpecificModel(BaseEmbeddingModel):
    """Custom model fine-tuned for SDG content"""
    
    def __init__(self, model_path: str = "bert-base-multilingual-cased"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.dimension = self.model.config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Loaded custom SDG model: {model_path}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 16,
               max_length: int = 512,
               **kwargs) -> np.ndarray:
        """Generate custom SDG embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            all_embeddings.extend(embeddings.cpu().numpy())
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        return self.dimension

class EmbeddingManager:
    """
    Manager for multiple embedding models with SDG-specific optimizations
    Enhanced version of your text_vektorizer.py functionality
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models: Dict[str, BaseEmbeddingModel] = {}
        self.default_model = "sentence_transformer"
        
        # Initialize default models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available embedding models"""
        try:
            # Sentence Transformer (multilingual)
            self.models["sentence_transformer"] = SentenceTransformerModel(
                self.config.get("sentence_transformer_model", "paraphrase-multilingual-MiniLM-L12-v2")
            )
            
            # OpenAI (if API key provided)
            if self.config.get("openai_api_key"):
                self.models["openai"] = OpenAIEmbeddingModel(
                    api_key=self.config.get("openai_api_key")
                )
            
            # Custom SDG model
            self.models["sdg_custom"] = SDGSpecificModel(
                self.config.get("custom_model_path", "bert-base-multilingual-cased")
            )
            
            logger.info(f"Initialized {len(self.models)} embedding models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def encode(self, 
               texts: Union[str, List[str]], 
               model_name: str = None,
               **kwargs) -> np.ndarray:
        """Generate embeddings using specified model"""
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available: {list(self.models.keys())}")
        
        return self.models[model_name].encode(texts, **kwargs)
    
    def encode_sdg_content(self, 
                          content: str,
                          sdg_goals: List[int] = None,
                          language: str = "en",
                          **kwargs) -> Dict[str, Any]:
        """
        Enhanced SDG-specific encoding with metadata
        Integrates your SDG classification from keywords.py
        """
        # Generate base embeddings
        embeddings = self.encode(content, **kwargs)
        
        # Add SDG-specific metadata
        metadata = {
            "embedding": embeddings[0] if len(embeddings) == 1 else embeddings,
            "content_length": len(content),
            "language": language,
            "timestamp": np.datetime64('now'),
            "model_used": kwargs.get("model_name", self.default_model),
            "dimension": len(embeddings) if len(embeddings) > 0 else 0
        }
        
        # Add SDG classification if provided
        if sdg_goals:
            metadata["sdg_goals"] = sdg_goals
            metadata["primary_sdg"] = sdg_goals if sdg_goals else None
        
        return metadata
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about embedding model"""
        model_name = model_name or self.default_model
        model = self.models.get(model_name)
        
        if not model:
            return {}
        
        return {
            "model_name": model_name,
            "dimension": model.get_dimension(),
            "type": type(model).__name__,
            "available": True
        }
    
    def batch_encode_with_progress(self, 
                                 texts: List[str], 
                                 batch_size: int = 100,
                                 model_name: str = None) -> List[np.ndarray]:
        """Batch encoding with progress tracking"""
        model_name = model_name or self.default_model
        results = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, model_name=model_name)
            results.extend(batch_embeddings)
            
            batch_num = (i // batch_size) + 1
            logger.info(f"Completed batch {batch_num}/{total_batches}")
        
        return results

# Language-specific embedding configurations
LANGUAGE_MODEL_MAPPING = {
    "en": "paraphrase-multilingual-MiniLM-L12-v2",
    "de": "paraphrase-multilingual-MiniLM-L12-v2", 
    "fr": "paraphrase-multilingual-MiniLM-L12-v2",
    "es": "paraphrase-multilingual-MiniLM-L12-v2",
    "zh": "paraphrase-multilingual-MiniLM-L12-v2",
    "hi": "paraphrase-multilingual-MiniLM-L12-v2"
}

def get_optimal_model_for_language(language: str) -> str:
    """Get optimal embedding model for specific language"""
    return LANGUAGE_MODEL_MAPPING.get(language, "paraphrase-multilingual-MiniLM-L12-v2")
