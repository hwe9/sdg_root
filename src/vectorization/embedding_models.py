
import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import torch.nn.functional as F
from abc import ABC
from abc import abstractmethod

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
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                 cache_dir: Optional[str] = None,
                 local_files_only: bool = False):
        import os
        self.model_name = model_name
        self.cache_dir = cache_dir or os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "/cache/huggingface"
        self.local_files_only = local_files_only or os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
        
        # Set cache directory for HuggingFace if provided
        if self.cache_dir:
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            logger.info(f"Using HuggingFace cache directory: {self.cache_dir}")
        
        # Determine local model path if available (for fully offline mode)
        resolved_model_path = self._resolve_local_model_path(self.model_name)
        model_reference = resolved_model_path or self.model_name
        if resolved_model_path:
            logger.info(f"Found local model snapshot for {self.model_name}: {resolved_model_path}")
        
        # Support offline mode if local_files_only is enabled
        try:
            if self.local_files_only:
                logger.info(f"Loading model in offline mode (local files only): {model_reference}")
                self.model = SentenceTransformer(
                    model_reference,
                    cache_folder=self.cache_dir,
                    local_files_only=True
                )
            else:
                self.model = SentenceTransformer(
                    model_reference,
                    cache_folder=self.cache_dir
                )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Loaded SentenceTransformer model: {self.model_name} (dimension: {self.dimension})")
        except Exception as e:
            # If online load fails and cache_dir is set, try offline snapshot
            if not resolved_model_path:
                logger.warning(f"⚠️ Primary load failed, attempting to locate local snapshot: {e}")
                snapshot_path = self._resolve_local_model_path(self.model_name)
            else:
                snapshot_path = resolved_model_path
            if snapshot_path and not self.local_files_only:
                try:
                    logger.info(f"Retrying with local snapshot path: {snapshot_path}")
                    self.model = SentenceTransformer(
                        snapshot_path,
                        cache_folder=self.cache_dir,
                        local_files_only=True
                    )
                    self.dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(f"✅ Loaded model from local snapshot: {snapshot_path}")
                    return
                except Exception as offline_error:
                    logger.error(f"❌ Failed to load model even from local snapshot: {offline_error}")
                    raise
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _resolve_local_model_path(self, model_name: str) -> Optional[str]:
        """Try to resolve a local snapshot directory for the given model."""
        import os
        import glob
        if not self.cache_dir:
            logger.debug(f"No cache_dir set for model resolution")
            return None
        sanitized = model_name.replace("/", "--")
        hub_root = os.path.join(self.cache_dir, "hub")
        logger.debug(f"Resolving model path for '{model_name}' (sanitized: '{sanitized}') in {hub_root}")
        
        if not os.path.isdir(hub_root):
            logger.warning(f"Hub root directory does not exist: {hub_root}")
            return None
            
        candidates: list[str] = []
        # Exact match
        exact_pattern = os.path.join(hub_root, f"models--{sanitized}")
        exact_matches = glob.glob(exact_pattern)
        candidates.extend(exact_matches)
        logger.debug(f"Exact pattern matches: {exact_matches}")
        
        # Models downloaded under fully qualified names (e.g. sentence-transformers/...)
        if not candidates:
            pattern = os.path.join(hub_root, f"models--*{sanitized}*")
            pattern_matches = glob.glob(pattern)
            candidates.extend(pattern_matches)
            logger.debug(f"Pattern '{pattern}' matches: {pattern_matches}")
        
        logger.debug(f"Total candidates found: {len(candidates)}")
        for hub_base in candidates:
            snapshots_dir = os.path.join(hub_base, "snapshots")
            # Preferred: use ref from refs/main
            ref_file = os.path.join(hub_base, "refs", "main")
            if os.path.exists(ref_file):
                try:
                    with open(ref_file, "r", encoding="utf-8") as f:
                        ref = f.read().strip()
                    candidate = os.path.join(snapshots_dir, ref)
                    if os.path.isdir(candidate):
                        return candidate
                except Exception as e:
                    logger.debug(f"Unable to read refs for {model_name} at {hub_base}: {e}")
            if os.path.isdir(snapshots_dir):
                snapshot_candidates = sorted(glob.glob(os.path.join(snapshots_dir, "*")))
                for candidate in snapshot_candidates:
                    if os.path.isdir(candidate):
                        return candidate
        # Last resort: check direct folder (used when copying manually)
        direct_path = os.path.join(self.cache_dir, model_name)
        if os.path.isdir(direct_path):
            return direct_path
        return None
    
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
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        self.model_name = model_name
        # Dimensions: small=1536, large=3072, ada-002=1536
        if model_name in ["text-embedding-3-small", "text-embedding-ada-002"]:
            self.dimension = 1536
        elif model_name in ["text-embedding-3-large"]:
            self.dimension = 3072
        else:
            self.dimension = 1536
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info(f"Initialized OpenAI embedding model (v1 client): {model_name}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        
        if isinstance(texts, str):
                texts = [texts]
        if not isinstance(texts, list) or len(texts) == 0:
                raise ValueError("texts must be a non-empty list or a string")
        try:
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            vecs = [d.embedding for d in resp.data]
            return np.asarray(vecs, dtype=float)
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
        
        try:
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

                    del embeddings, model_output, encoded_input
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            # Cleanup auch bei Fehlern
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
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
            import os
            # Sentence Transformer (multilingual)
            model_name = self.config.get("sentence_transformer_model", "paraphrase-multilingual-MiniLM-L12-v2")
            cache_dir = self.config.get("cache_dir") or os.environ.get("HF_CACHE_DIR")
            local_files_only = self.config.get("local_files_only", False) or os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
            
            self.models["sentence_transformer"] = SentenceTransformerModel(
                model_name=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only
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
