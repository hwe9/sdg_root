"""
SDG Vectorization Service
Handles embedding generation, vector storage, and semantic search
"""
__version__ = "2.0.0"
__author__ = "SDG Pipeline Team"

from .embedding_models import EmbeddingManager
from .vector_db_client import VectorDBClient
from .similarity_search import SimilaritySearch

__all__ = ["EmbeddingManager", "VectorDBClient", "SimilaritySearch"]
