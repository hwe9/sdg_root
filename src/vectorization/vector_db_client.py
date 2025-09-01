"""
Vector Database Client for SDG Pipeline
Enhanced Weaviate integration with connection pooling and SDG schema
"""
import logging
import time
from typing import List, Dict, Any, Optional, Union
import weaviate
import numpy as np
from weaviate.embedded import EmbeddedOptions
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBClient:
    """
    Enhanced Weaviate client for SDG vector operations
    Includes connection pooling, retry logic, and SDG-specific schema
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self.max_connections = config.get("max_connections", 10)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # SDG-specific configuration
        self.sdg_classes = [
            "SDGArticle", "SDGProgress", "SDGTarget", 
            "SDGIndicator", "RegionalData", "AITopic"
        ]
        
        self._initialize_client()
        self._setup_sdg_schema()
    
    def _initialize_client(self):
        """Initialize Weaviate client with configuration and URL validation"""
        try:
            if self.config.get("embedded", False):
                # Embedded Weaviate for development
                self.client = weaviate.Client(
                    embedded_options=EmbeddedOptions(
                        hostname=self.config.get("hostname", "localhost"),
                        port=self.config.get("port", 8080),
                        grpc_port=self.config.get("grpc_port", 50051)
                    )
                )
            else:
                # Remote Weaviate instance mit URL-Validierung
                weaviate_url = self.config.get("url", "http://localhost:8080")
                
                # URL-Validierung hinzufügen
                if not weaviate_url.startswith(('http://', 'https://')):
                    raise ValueError(f"Invalid Weaviate URL format: {weaviate_url}")
                
                auth_config = None
                if self.config.get("api_key"):
                    auth_config = weaviate.AuthApiKey(api_key=self.config["api_key"])
                
                headers = {}
                if self.config.get("openai_api_key"):
                    headers["X-OpenAI-Api-Key"] = self.config["openai_api_key"]
                
                self.client = weaviate.Client(
                    url=weaviate_url,  # Validierte URL verwenden
                    auth_client_secret=auth_config,
                    additional_headers=headers
                )
            
            # Test connection mit Retry-Logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.client.is_ready():
                        logger.info(f"Weaviate client initialized successfully (attempt {attempt + 1})")
                        break
                    else:
                        raise ConnectionError(f"Weaviate client not ready (attempt {attempt + 1})")
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to establish Weaviate connection after {max_retries} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Weaviate connection attempt {attempt + 1} failed: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            logger.error(f"Error initializing Weaviate client: {e}")
            raise
    
    def _setup_sdg_schema(self):
        """Setup SDG schema with version compatibility check"""
        try:
            # Schema-Versions-Kompatibilität prüfen
            current_version = "1.0"
            
            try:
                existing_schema = self.client.schema.get("SDGArticle")
                existing_version = existing_schema.get("version", "0.0")
                
                if existing_version != current_version:
                    logger.warning(f"Schema version mismatch: {existing_version} != {current_version}")
                    # Schema-Migration falls erforderlich
                    self._migrate_schema(existing_version, current_version)
                else:
                    logger.info("SDGArticle schema already exists with correct version")
                    return
                    
            except weaviate.exceptions.UnexpectedStatusCodeException:
                # Schema existiert nicht, neu erstellen
                pass
            
            # Verbesserte Schema-Definition
            sdg_article_schema = {
                "class": "SDGArticle",
                "description": "SDG-related articles and documents with full metadata",
                "vectorizer": "none",
                "version": current_version,
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Article title",
                        "tokenization": "word",
                        "indexFilterable": True
                    },
                    {
                        "name": "content", 
                        "dataType": ["text"],
                        "description": "Full article content",
                        "tokenization": "word",
                        "indexSearchable": True
                    },
                    {
                        "name": "summary",
                        "dataType": ["text"], 
                        "description": "Article summary",
                        "tokenization": "word"
                    },
                    {
                        "name": "sdg_goals",
                        "dataType": ["int[]"],
                        "description": "Related SDG goals (1-17)",
                        "indexFilterable": True
                    },
                    {
                        "name": "region",
                        "dataType": ["text"],
                        "description": "Geographic region",
                        "indexFilterable": True
                    },
                    {
                        "name": "language", 
                        "dataType": ["text"],
                        "description": "Content language",
                        "indexFilterable": True
                    },
                    {
                        "name": "confidence_score",
                        "dataType": ["number"],
                        "description": "SDG classification confidence (0.0-1.0)",
                        "indexFilterable": True,
                        "indexSearchable": False
                    },
                    {
                        "name": "publication_date",
                        "dataType": ["date"],
                        "description": "Publication date",
                        "indexFilterable": True
                    },
                    {
                        "name": "source_url",
                        "dataType": ["text"],
                        "description": "Original source URL",
                        "indexFilterable": False
                    }
                ]
            }
            
            self.client.schema.create_class(sdg_article_schema)
            logger.info("Created new SDGArticle schema with version validation")
            
        except Exception as e:
            logger.error(f"Error setting up SDG schema: {e}")
            self._create_fallback_schema()

    def _migrate_schema(self, old_version: str, new_version: str):
        """Handle schema migration between versions"""
        logger.info(f"Migrating schema from {old_version} to {new_version}")
        
        # Backup existing data
        try:
            backup_data = self.client.query.get("SDGArticle").do()
            logger.info(f"Backed up {len(backup_data.get('data', {}).get('Get', {}).get('SDGArticle', []))} documents")
        except Exception as e:
            logger.error(f"Schema migration backup failed: {e}")
            raise
        
        # Delete and recreate schema
        try:
            self.client.schema.delete_class("SDGArticle")
            logger.info("Deleted old schema")
        except Exception as e:
            logger.warning(f"Could not delete old schema: {e}")

    def _create_fallback_schema(self):
        """Create simple fallback schema"""
        simple_schema = {
            "class": "SDGArticle",
            "vectorizer": "none",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "sdg_goals", "dataType": ["int[]"]}
            ]
        }
        try:
            self.client.schema.create_class(simple_schema)
            logger.info("Created fallback SDGArticle schema")
        except Exception as e:
            logger.error(f"Failed to create fallback schema: {e}")
        
    def _create_additional_schemas(self):
        """Create additional SDG-related schemas"""
        schemas = [
            {
                "class": "SDGProgress",
                "description": "SDG progress tracking data",
                "vectorizer": "none",
                "properties": [
                    {"name": "country", "dataType": ["text"]},
                    {"name": "sdg_goal", "dataType": ["int"]},
                    {"name": "indicator_value", "dataType": ["number"]},
                    {"name": "year", "dataType": ["int"]},
                    {"name": "data_source", "dataType": ["text"]}
                ]
            },
            {
                "class": "RegionalData", 
                "description": "Region-specific SDG data",
                "vectorizer": "none",
                "properties": [
                    {"name": "region", "dataType": ["text"]},
                    {"name": "country", "dataType": ["text"]},
                    {"name": "sdg_scores", "dataType": ["number[]"]},
                    {"name": "metadata", "dataType": ["text"]}
                ]
            }
        ]
        
        for schema in schemas:
            try:
                if not self.client.schema.exists(schema["class"]):
                    self.client.schema.create_class(schema)
                    logger.info(f"Created {schema['class']} schema")
            except Exception as e:
                logger.error(f"Failed to create {schema['class']} schema: {e}")
    
    async def insert_embeddings(self, 
                              documents: List[Dict[str, Any]], 
                              class_name: str = "SDGArticle",
                              batch_size: int = 100) -> List[str]:
        """
        Insert documents with embeddings into Weaviate
        """
        uuids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(documents)} documents in {total_batches} batches")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_uuids = await self._insert_batch(batch, class_name)
            uuids.extend(batch_uuids)
            
            batch_num = (i // batch_size) + 1
            logger.info(f"Inserted batch {batch_num}/{total_batches}")
        
        return uuids
    
    async def _insert_batch(self, documents: List[Dict[str, Any]], class_name: str) -> List[str]:
        """Insert a batch of documents with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                with self.client.batch as batch:
                    batch.batch_size = len(documents)
                    uuids = []
                    
                    for doc in documents:
                        # Extract vector and properties
                        vector = doc.pop("vector", None)
                        uuid = self.client.batch.add_data_object(
                            data_object=doc,
                            class_name=class_name,
                            vector=vector
                        )
                        uuids.append(uuid)
                    
                    return uuids
                    
            except Exception as e:
                logger.warning(f"Batch insert attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
    
    def search_similar(self, 
                      query_vector: np.ndarray,
                      class_name: str = "SDGArticle", 
                      limit: int = 10,
                      where_filter: Dict[str, Any] = None,
                      additional_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Semantic similarity search with SDG-specific filtering
        """
        try:
            # Build the query
            query_builder = (
                self.client.query
                .get(class_name, additional_fields or ["title", "summary", "sdg_goals", "region"])
                .with_near_vector({
                    "vector": query_vector.tolist(),
                    "certainty": 0.7
                })
                .with_limit(limit)
                .with_additional(["certainty", "distance"])
            )
            
            # Add where filter if provided
            if where_filter:
                query_builder = query_builder.with_where(where_filter)
            
            result = query_builder.do()
            
            # Extract results
            class_results = result.get("data", {}).get("Get", {}).get(class_name, [])
            return class_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def search_by_sdg_goals(self, 
                           sdg_goals: List[int],
                           query_vector: np.ndarray = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Search for content related to specific SDG goals"""
        where_filter = {
            "operator": "ContainsAny",
            "path": ["sdg_goals"], 
            "valueIntArray": sdg_goals
        }
        
        if query_vector is not None:
            return self.search_similar(
                query_vector=query_vector,
                where_filter=where_filter,
                limit=limit
            )
        else:
            # Pure filter search without vector similarity
            try:
                result = (
                    self.client.query
                    .get("SDGArticle", ["title", "summary", "sdg_goals", "region", "confidence_score"])
                    .with_where(where_filter)
                    .with_limit(limit)
                    .do()
                )
                return result.get("data", {}).get("Get", {}).get("SDGArticle", [])
            except Exception as e:
                logger.error(f"Error in SDG goal search: {e}")
                raise
    
    def search_by_region(self, 
                        region: str,
                        query_vector: np.ndarray = None,
                        limit: int = 30) -> List[Dict[str, Any]]:
        """Search for region-specific SDG content"""
        where_filter = {
            "operator": "Equal",
            "path": ["region"],
            "valueText": region
        }
        
        if query_vector is not None:
            return self.search_similar(
                query_vector=query_vector,
                where_filter=where_filter,
                limit=limit
            )
        else:
            try:
                result = (
                    self.client.query
                    .get("SDGArticle", ["title", "summary", "region", "sdg_goals"])
                    .with_where(where_filter)
                    .with_limit(limit)
                    .do()
                )
                return result.get("data", {}).get("Get", {}).get("SDGArticle", [])
            except Exception as e:
                logger.error(f"Error in region search: {e}")
                raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            for class_name in self.sdg_classes:
                try:
                    if self.client.schema.exists(class_name):
                        result = self.client.query.aggregate(class_name).with_meta_count().do()
                        count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
                        stats[class_name] = count
                except Exception as e:
                    logger.warning(f"Could not get statistics for {class_name}: {e}")
                    stats[class_name] = 0
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector database"""
        try:
            is_ready = self.client.is_ready()
            is_live = self.client.is_live()
            
            return {
                "status": "healthy" if is_ready and is_live else "unhealthy",
                "ready": is_ready,
                "live": is_live,
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": self.get_statistics()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def close(self):
        """Close the vector database connection"""
        if self.client:
            # Weaviate client doesn't have explicit close method
            # but we can clear the connection pool
            with self._pool_lock:
                self._connection_pool.clear()
            logger.info("Vector database client closed")

# Connection manager for async operations
@asynccontextmanager
async def get_vector_client(config: Dict[str, Any]):
    """Context manager for vector database operations"""
    client = VectorDBClient(config)
    try:
        yield client
    finally:
        client.close()
