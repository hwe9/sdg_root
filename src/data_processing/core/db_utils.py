import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import json
import weaviate
from contextlib import contextmanager
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import OperationalError, IntegrityError, DisconnectionError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Dict, Any, List, Optional, Union
try:
    from .secrets_manager import secrets_manager
except ImportError:
    from ..core.secrets_manager import secrets_manager
import logging
from datetime import datetime, timedelta
from ...core.dependency_manager import get_dependency_manager
import httpx
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL with fallback"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set, using default")
        return "postgresql://postgres:postgres@database_service:5432/sdg_pipeline"
    return db_url

DATABASE_URL = get_database_url()
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080")

def get_database_engine():
    """Create robust database engine with dependency validation"""
    # Import dependency_manager as specified in requirements
    from ...core.dependency_manager import dependency_manager
    
    # Check if we should wait for dependencies
    if hasattr(dependency_manager, '_startup_complete') and not dependency_manager._startup_complete.is_set():
        logger.info("Waiting for database dependencies...")
        # In production, you might want to implement a sync wait here
    
    engine_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
        "poolclass": QueuePool,
        "connect_args": {
            "connect_timeout": 30,
            "application_name": "SDG_Pipeline"
        }
    }
    
    engine = create_engine(DATABASE_URL, **engine_kwargs)
    
    # Test connection with dependency awareness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            
            # Update dependency status if manager is available
            if hasattr(dependency_manager, 'service_status') and 'database' in dependency_manager.service_status:
                from ...core.dependency_manager import ServiceStatus
                dependency_manager.service_status['database'] = ServiceStatus.HEALTHY
            
            break
        except (OperationalError, DisconnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to establish database connection after {max_retries} attempts")
                # Update dependency status to failed
                if hasattr(dependency_manager, 'service_status') and 'database' in dependency_manager.service_status:
                    from ...core.dependency_manager import ServiceStatus
                    dependency_manager.service_status['database'] = ServiceStatus.FAILED
                raise
    
    return engine

def check_database_health() -> bool:
    """Check database health with dependency status update"""
    from ...core.dependency_manager import dependency_manager
    
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Update dependency status
        if hasattr(dependency_manager, 'service_status') and 'database' in dependency_manager.service_status:
            from ...core.dependency_manager import ServiceStatus
            dependency_manager.service_status['database'] = ServiceStatus.HEALTHY
        
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        
        # Update dependency status
        if hasattr(dependency_manager, 'service_status') and 'database' in dependency_manager.service_status:
            from ...core.dependency_manager import ServiceStatus
            dependency_manager.service_status['database'] = ServiceStatus.UNHEALTHY
        
        return False

def get_database_session():
    """Get database session via dependency manager"""
    dep_manager = get_dependency_manager()
    return dep_manager.get_database_session()

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic retry"""
    engine = get_database_engine()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with engine.begin() as connection:
                yield connection
                break
        except (OperationalError, DisconnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database operation attempt {attempt + 1} failed: {e}")
                time.sleep(1)
            else:
                logger.error("Database operation failed after retries")
                raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise

def _check_weaviate_health(base_url: str, timeout: float = 5.0) -> bool:
    """
    Check Weaviate readiness using /.well-known/ready or /v1/.well-known/ready,
    with /v1/meta as a fallback for older healthcheck styles.
    """
    base = base_url.rstrip("/")
    endpoints = ["/.well-known/ready", "/v1/.well-known/ready", "/v1/meta"]
    for path in endpoints:
        url = f"{base}{path}"
        try:
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True
        except Exception:
            continue
    return False

def get_weaviate_client():
    """Create and return Weaviate client instance with explicit health check"""
    if not _check_weaviate_health(WEAVIATE_URL):
        raise ConnectionError(
            f"Weaviate not ready at {WEAVIATE_URL}; readiness check failed (/.well-known/ready or /v1/meta)"
        )
    try:
        client = weaviate.Client(url=WEAVIATE_URL)

        # Ensure schema class exists (Schema-Erzeugung beibehalten)
        try:
            client.schema.get("ArticleVector")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            _create_weaviate_schema(client)

        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise ConnectionError(f"Weaviate connection failed: {e}")


def _create_weaviate_schema(client):
    class_obj = {
        "class": "ArticleVector",
        "description": "SDG article embeddings for semantic search",
        "vectorizer": "none", 
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Article text content"
            },
            {
                "name": "articleId", 
                "dataType": ["int"],
                "description": "Reference to article ID in PostgreSQL"
            },
            {
                "name": "chunkId",
                "dataType": ["int"], 
                "description": "Chunk ID for long documents"
            },
            {
                "name": "sdgGoals",
                "dataType": ["int[]"],
                "description": "Related SDG goals"
            },
            {
                "name": "language",
                "dataType": ["text"],
                "description": "Content language"
            },
            {
                "name": "region",
                "dataType": ["text"],
                "description": "Geographic region"
            }
        ]
    }
    
    client.schema.create_class(class_obj)
    logger.info("Created ArticleVector schema in Weaviate")

async def save_to_database(metadata: Dict[str, Any], text_content: str, embeddings: List[float], chunks_data: Optional[List[Dict]] = None):
    """Save to database using centralized dependency management"""
    article_id = None
    
    try:
        dep_manager = get_dependency_manager()
        async with dep_manager.get_db_connection() as connection:
            article_id = _insert_article(connection, metadata, text_content)

            if chunks_data:
                _insert_article_chunks(connection, article_id, chunks_data)

            _insert_tag_relationships(connection, article_id, metadata)
            _insert_ai_topic_relationships(connection, article_id, metadata)
            _insert_sdg_target_relationships(connection, article_id, metadata)

            logger.info(f"✅ Article {article_id} saved to PostgreSQL successfully")

        # Vector-Client wie gehabt über den Dependency-Manager
        async with dep_manager.get_vector_client() as vector_client:
            _save_to_weaviate_via_client(vector_client, article_id, text_content, embeddings, chunks_data, metadata)

        return article_id
        
    except Exception as e:
        logger.error(f"❌ Error saving to database: {e}")
        if article_id:
            logger.error(f"Article ID {article_id} may be partially saved")
        raise


def _save_to_weaviate_via_client(vector_client, article_id: int, text_content: str, embeddings: List[float], 
                                chunks_data: Optional[List[Dict]] = None, metadata: Dict[str, Any] = None):
    """Save embeddings to Weaviate using dependency-managed client"""
    try:
        if chunks_data and len(chunks_data) > 0:
            # Save each chunk as separate vector
            for i, chunk_data in enumerate(chunks_data):
                chunk_vector = chunk_data.get("embedding") or embeddings
                
                vector_data = {
                    "text": chunk_data["text"],
                    "articleId": article_id,
                    "chunkId": i,
                    "sdgGoals": metadata.get('sdg_goals', []) if metadata else [],
                    "language": metadata.get('language', 'en') if metadata else 'en',
                    "region": metadata.get('region', '') if metadata else ''
                }
                
                vector_client.insert_data(vector_data, "ArticleVector", chunk_vector)
        else:
            # Save full document as single vector
            vector_data = {
                "text": text_content,
                "articleId": article_id,
                "chunkId": 0,
                "sdgGoals": metadata.get('sdg_goals', []) if metadata else [],
                "language": metadata.get('language', 'en') if metadata else 'en',
                "region": metadata.get('region', '') if metadata else ''
            }
            
            vector_client.insert_data(vector_data, "ArticleVector", embeddings)
        
        logger.info(f"✅ Vector data for article {article_id} saved via dependency manager")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Weaviate via client: {e}")
        raise


def _insert_article(connection, metadata: Dict[str, Any], text_content: str) -> int:
    
    publication_year = _extract_publication_year(metadata)
    
    insert_query = text("""
        INSERT INTO articles (
            title, content_original, content_english, summary, keywords, sdg_id, 
            authors, publication_year, publication_date, publisher, doi, isbn, 
            region, country_code, language, context, study_type, research_methods, 
            data_sources, funding, funding_info, bias_indicators, abstract_original, 
            abstract_english, relevance_questions, source_url, availability, 
            citation_count, impact_metrics, impact_factor, policy_impact,
            word_count, content_quality_score, created_at
        ) VALUES (
            :title, :content_original, :content_english, :summary, :keywords, :sdg_id,
            :authors, :publication_year, :publication_date, :publisher, :doi, :isbn,
            :region, :country_code, :language, :context, :study_type, :research_methods,
            :data_sources, :funding, :funding_info, :bias_indicators, :abstract_original,
            :abstract_english, :relevance_questions, :source_url, :availability,
            :citation_count, :impact_metrics, :impact_factor, :policy_impact,
            :word_count, :content_quality_score, NOW()
        ) RETURNING id
    """)
    
    content_quality_score = _calculate_content_quality_score(metadata, text_content)
    
    params = {
        "title": metadata.get('title', 'Untitled')[:500],  
        "content_original": text_content,
        "content_english": metadata.get('content_english') or text_content,
        "summary": metadata.get('summary'),
        "keywords": metadata.get('keywords'),
        "sdg_id": metadata.get('sdg_id'),
        "authors": metadata.get('authors'),
        "publication_year": publication_year,
        "publication_date": _parse_publication_date(metadata.get('publication_date')),
        "publisher": metadata.get('publisher'),
        "doi": metadata.get('doi'),
        "isbn": metadata.get('isbn'),
        "region": metadata.get('region'),
        "country_code": metadata.get('country_code'),
        "language": metadata.get('language', 'en'),
        "context": metadata.get('context'),
        "study_type": metadata.get('study_type'),
        "research_methods": metadata.get('research_methods'),
        "data_sources": metadata.get('data_sources'),
        "funding": metadata.get('funding'),
        "funding_info": metadata.get('funding_info'),
        "bias_indicators": metadata.get('bias_indicators'),
        "abstract_original": metadata.get('abstract_original'),
        "abstract_english": metadata.get('abstract_english'),
        "relevance_questions": metadata.get('relevance_questions'),
        "source_url": metadata.get('source_url'),
        "availability": metadata.get('availability'),
        "citation_count": metadata.get('citation_count', 0),
        "impact_metrics": json.dumps(metadata.get('impact_metrics')) if metadata.get('impact_metrics') else None,
        "impact_factor": metadata.get('impact_factor'),
        "policy_impact": metadata.get('policy_impact'),
        "word_count": len(text_content.split()) if text_content else 0,
        "content_quality_score": content_quality_score
    }
    
    result = connection.execute(insert_query, params)
    return result.scalar_one()

def _insert_article_chunks(connection, article_id: int, chunks_data: List[Dict]):

    chunk_query = text("""
        INSERT INTO article_chunks (
            article_id, chunk_id, chunk_order, text, chunk_length, 
            sdg_section, sub_section_id, sdg_relevance_scores, 
            confidence_score, created_at
        ) VALUES (
            :article_id, :chunk_id, :chunk_order, :text, :chunk_length,
            :sdg_section, :sub_section_id, :sdg_relevance_scores,
            :confidence_score, NOW()
        )
    """)
    
    for i, chunk_data in enumerate(chunks_data):
        chunk_params = {
            "article_id": article_id,
            "chunk_id": chunk_data.get("chunk_id", i),
            "chunk_order": i,
            "text": chunk_data["text"],
            "chunk_length": len(chunk_data["text"]),
            "sdg_section": chunk_data.get("sdg_section", "general"),
            "sub_section_id": chunk_data.get("sub_section_id"),
            "sdg_relevance_scores": json.dumps(chunk_data.get("sdg_relevance_scores")) if chunk_data.get("sdg_relevance_scores") else None,
            "confidence_score": chunk_data.get("confidence_score", 0.0)
        }
        
        connection.execute(chunk_query, chunk_params)
    
    logger.info(f"Inserted {len(chunks_data)} chunks for article {article_id}")

def _insert_tag_relationships(connection, article_id: int, metadata: Dict[str, Any]):
    
    tags = metadata.get('tags', [])
    if not tags:
        return
    
    for tag_name in tags:
        if not tag_name or not tag_name.strip():
            continue
            
        tag_name = tag_name.strip()[:100]  
        
        tag_id = connection.execute(
            text("SELECT id FROM tags WHERE name = :name"),
            {"name": tag_name}
        ).scalar_one_or_none()
        
        if not tag_id:
            tag_id = connection.execute(
                text("""
                    INSERT INTO tags (name, category, usage_count) 
                    VALUES (:name, :category, 1) 
                    RETURNING id
                """),
                {"name": tag_name, "category": "general"}
            ).scalar_one()
        else:
            connection.execute(
                text("UPDATE tags SET usage_count = usage_count + 1 WHERE id = :tag_id"),
                {"tag_id": tag_id}
            )
        
        try:
            connection.execute(
                text("""
                    INSERT INTO articles_tags (article_id, tag_id) 
                    VALUES (:article_id, :tag_id)
                    ON CONFLICT (article_id, tag_id) DO NOTHING
                """),
                {"article_id": article_id, "tag_id": tag_id}
            )
        except IntegrityError:
            pass 

def _insert_ai_topic_relationships(connection, article_id: int, metadata: Dict[str, Any]):
    """Insert article-AI topic relationships"""
    ai_topics = metadata.get('ai_topics', [])
    if not ai_topics:
        return
    
    for topic_name in ai_topics:
        if not topic_name or not topic_name.strip():
            continue
            
        topic_name = topic_name.strip()[:100]
        
        topic_id = connection.execute(
            text("SELECT id FROM ai_topics WHERE name = :name"),
            {"name": topic_name}
        ).scalar_one_or_none()
        
        if not topic_id:
            topic_id = connection.execute(
                text("""
                    INSERT INTO ai_topics (name, category, created_at) 
                    VALUES (:name, :category, NOW()) 
                    RETURNING id
                """),
                {"name": topic_name, "category": "general"}
            ).scalar_one()
        
        try:
            connection.execute(
                text("""
                    INSERT INTO articles_ai_topics (article_id, ai_topic_id) 
                    VALUES (:article_id, :topic_id)
                    ON CONFLICT (article_id, ai_topic_id) DO NOTHING
                """),
                {"article_id": article_id, "topic_id": topic_id}
            )
        except IntegrityError:
            pass

def _insert_sdg_target_relationships(connection, article_id: int, metadata: Dict[str, Any]):
    """Insert article-SDG target relationships"""
    sdg_goals = metadata.get('sdg_goals', [])
    if not sdg_goals:
        return
    
    for goal_id in sdg_goals:
        sdg_ids = []
    
    # Primary SDG from sdg_id field
    if metadata.get('sdg_id'):
        sdg_ids.append(metadata['sdg_id'])
    
    # Additional SDGs from sdg_goals list
    if metadata.get('sdg_goals'):
        if isinstance(metadata['sdg_goals'], list):
            sdg_ids.extend(metadata['sdg_goals'])
        elif isinstance(metadata['sdg_goals'], int):
            sdg_ids.append(metadata['sdg_goals'])
    
    # Remove duplicates and ensure valid range
    sdg_ids = list(set([sdg for sdg in sdg_ids if isinstance(sdg, int) and 1 <= sdg <= 17]))
    
    for sdg_id in sdg_ids:
        confidence_score = metadata.get('sdg_confidence', 0.8)  # Default confidence
        
        connection.execute(
            text("""
                INSERT INTO articles_sdg_targets (article_id, sdg_id, confidence_score) 
                VALUES (:article_id, :sdg_id, :confidence_score) 
                ON CONFLICT (article_id, sdg_id) DO UPDATE SET confidence_score = :confidence_score
            """),
            {
                "article_id": article_id,
                "sdg_id": sdg_id,
                "confidence_score": confidence_score
            }
        )

def _save_to_weaviate(article_id: int, text_content: str, embeddings: List[float], 
                     chunks_data: Optional[List[Dict]] = None, metadata: Dict[str, Any] = None):
    """Save embeddings to Weaviate vector database"""
    try:
        client = get_weaviate_client()
        
        # Ensure ArticleVector schema exists
        try:
            client.schema.get("ArticleVector")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            _create_weaviate_schema(client)
        
        if chunks_data and len(chunks_data) > 0:
            # Save each chunk as separate vector
            for i, chunk_data in enumerate(chunks_data):
                chunk_vector = chunk_data.get("embedding") or embeddings
                
                vector_data = {
                    "text": chunk_data["text"],
                    "articleId": article_id,
                    "chunkId": i,
                    "sdgGoals": metadata.get('sdg_goals', []) if metadata else [],
                    "language": metadata.get('language', 'en') if metadata else 'en',
                    "region": metadata.get('region', '') if metadata else ''
                }
                
                client.data_object.create(
                    data_object=vector_data,
                    class_name="ArticleVector",
                    vector=chunk_vector
                )
        else:
            # Save full document as single vector
            vector_data = {
                "text": text_content,
                "articleId": article_id,
                "chunkId": 0,
                "sdgGoals": metadata.get('sdg_goals', []) if metadata else [],
                "language": metadata.get('language', 'en') if metadata else 'en',
                "region": metadata.get('region', '') if metadata else ''
            }
            
            client.data_object.create(
                data_object=vector_data,
                class_name="ArticleVector",
                vector=embeddings
            )
        
        logger.info(f"✅ Vector data for article {article_id} saved to Weaviate")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Weaviate: {e}")
        raise

def get_article_by_id(article_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve article by ID with related data"""
    engine = get_database_engine()
    
    try:
        with engine.connect() as connection:
            
            article_query = text("""
                SELECT a.*, s.name as sdg_name
                FROM articles a
                LEFT JOIN sdgs s ON a.sdg_id = s.id
                WHERE a.id = :article_id
            """)
            
            result = connection.execute(article_query, {"article_id": article_id}).fetchone()
            if not result:
                return None
            
            article = dict(result._mapping)
            
            chunks_query = text("""
                SELECT * FROM article_chunks 
                WHERE article_id = :article_id 
                ORDER BY chunk_order
            """)
            chunks = connection.execute(chunks_query, {"article_id": article_id}).fetchall()
            article["chunks"] = [dict(chunk._mapping) for chunk in chunks]
            
            tags_query = text("""
                SELECT t.name FROM tags t
                JOIN articles_tags at ON t.id = at.tag_id
                WHERE at.article_id = :article_id
            """)
            tags = connection.execute(tags_query, {"article_id": article_id}).fetchall()
            article["tags"] = [tag.name for tag in tags]
            
            return article
            
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        return None

def search_similar_content(
    query_embedding: List[float], 
    limit: int = 10,
    sdg_filter: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Search for similar content using Weaviate"""
    try:
        client = get_weaviate_client()
        
        query_builder = (
            client.query
            .get("ArticleVector", ["text", "articleId", "sdgGoals", "language", "region"])
            .with_near_vector({
                "vector": query_embedding,
                "certainty": 0.7
            })
            .with_limit(limit)
            .with_additional(["certainty", "distance"])
        )
        
        if sdg_filter:
            where_filter = {
                "operator": "ContainsAny",
                "path": ["sdgGoals"],
                "valueIntArray": sdg_filter
            }
            query_builder = query_builder.with_where(where_filter)
        
        result = query_builder.do()
        
        return result.get("data", {}).get("Get", {}).get("ArticleVector", [])
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return []

def _extract_publication_year(metadata: Dict[str, Any]) -> Optional[int]:
    """Extract publication year from various metadata fields"""
    import re
    year_fields = ['publication_year', 'year', 'published_year']
    
    for field in year_fields:
        year_value = metadata.get(field)
        if year_value:
            try:
                if isinstance(year_value, list) and len(year_value) > 0:
                    year_value = year_value[0]
                if isinstance(year_value, str):
                    year_match = re.match(r'(\d{4})', year_value)
                    if year_match:
                        return int(year_match.group(1))
                elif isinstance(year_value, int):
                    if 1900 <= year_value <= 2030:  # Reasonable year range
                        return year_value
            except (ValueError, TypeError):
                continue
    
    return None

def _parse_publication_date(date_string: str) -> Optional[datetime]:
    """Parse publication date from string"""
    if not date_string:
        return None
        
    try:
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_string), fmt)
            except ValueError:
                continue
                
        # If none of the formats work, try to extract year
        year_match = re.search(r'(\d{4})', str(date_string))
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2030:
                return datetime(year, 1, 1)
                
    except Exception as e:
        logger.warning(f"Could not parse date '{date_string}': {e}")
    
    return None

def _calculate_content_quality_score(metadata: Dict[str, Any], text_content: str) -> float:

    score = 0.0
    
    # Title quality (0-0.15)
    title = metadata.get('title', '')
    if title and len(title.strip()) > 10:
        score += 0.15
    elif title and len(title.strip()) > 5:
        score += 0.08
    
    # Content length (0-0.25)
    content_length = len(text_content.strip())
    if content_length > 5000:
        score += 0.25
    elif content_length > 2000:
        score += 0.20
    elif content_length > 500:
        score += 0.10
    elif content_length > 100:
        score += 0.05
    
    # Has authors (0-0.10)
    if metadata.get('authors'):
        score += 0.10
    
    # Has publication info (0-0.10)
    if metadata.get('publication_year') or metadata.get('publisher'):
        score += 0.10
    
    # Has DOI or ISBN (0-0.10)
    if metadata.get('doi') or metadata.get('isbn'):
        score += 0.10
    
    # Has abstract (0-0.10)
    if metadata.get('abstract_original') or metadata.get('abstract_english'):
        score += 0.10
    
    # Has source URL (0-0.05)
    if metadata.get('source_url'):
        score += 0.05
    
    # Has keywords or tags (0-0.05)
    if metadata.get('keywords') or metadata.get('tags'):
        score += 0.05
    
    # Has SDG classification (0-0.10)
    if metadata.get('sdg_id') or metadata.get('sdg_goals'):
        score += 0.10
    
    return min(score, 1.0)


def batch_save_to_database(items: List[Dict[str, Any]], batch_size: int = 50):
    """
    Batch save multiple items to database for better performance
    """
    engine = get_database_engine()
    successful_saves = 0
    failed_saves = 0
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        try:
            with engine.begin() as connection:
                for item in batch:
                    metadata = item.get('metadata', {})
                    text_content = item.get('text_content', '')
                    embeddings = item.get('embeddings', [])
                    chunks_data = item.get('chunks_data')
                    
                    try:
                        article_id = _insert_article(connection, metadata, text_content)
                        
                        if chunks_data:
                            _insert_article_chunks(connection, article_id, chunks_data)
                        
                        _insert_tag_relationships(connection, article_id, metadata)
                        _insert_ai_topic_relationships(connection, article_id, metadata)
                        _insert_sdg_target_relationships(connection, article_id, metadata)
                        
                        # Save to Weaviate in separate transaction
                        _save_to_weaviate(article_id, text_content, embeddings, chunks_data, metadata)
                        
                        successful_saves += 1
                        
                    except Exception as e:
                        logger.error(f"Error saving individual item: {e}")
                        failed_saves += 1
                        continue
                        
        except Exception as e:
            logger.error(f"Error in batch save: {e}")
            failed_saves += len(batch)
    
    logger.info(f"Batch save completed: {successful_saves} successful, {failed_saves} failed")
    return {"successful": successful_saves, "failed": failed_saves}

def cleanup_old_data(retention_days: int = 30):
    """
    Cleanup old data from database based on retention policy
    """
    engine = get_database_engine()
    
    try:
        with engine.begin() as connection:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Delete old articles (this will cascade to related tables)
            result = connection.execute(
                text("DELETE FROM articles WHERE created_at < :cutoff_date"),
                {"cutoff_date": cutoff_date}
            )
            
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old articles older than {retention_days} days")
            
            return deleted_count
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

def get_database_statistics() -> Dict[str, Any]:
    """Get database statistics"""
    engine = get_database_engine()
    
    try:
        with engine.connect() as connection:
            stats = {}
            
            article_stats = connection.execute(text("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(*) FILTER (WHERE has_embeddings = TRUE) as articles_with_embeddings,
                    AVG(content_quality_score) as avg_quality_score,
                    COUNT(DISTINCT language) as languages,
                    COUNT(DISTINCT region) as regions
                FROM articles
            """)).fetchone()
            
            stats.update(dict(article_stats._mapping))
            
            sdg_stats = connection.execute(text("""
                SELECT 
                    s.name,
                    COUNT(a.id) as article_count
                FROM sdgs s
                LEFT JOIN articles a ON s.id = a.sdg_id
                GROUP BY s.id, s.name
                ORDER BY article_count DESC
            """)).fetchall()
            
            stats["sdg_distribution"] = [dict(row._mapping) for row in sdg_stats]
            
            return stats
            
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {}

def get_database_health() -> Dict[str, Any]:
    """
    Check database health and return status information with dependency awareness
    """
    try:
        if check_database_health():
            engine = get_database_engine()
            
            with engine.connect() as connection:
                # Get table counts
                tables = ['articles', 'sdgs', 'actors', 'tags', 'ai_topics', 'article_chunks']
                table_counts = {}
                
                for table in tables:
                    try:
                        result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = result.scalar()
                    except Exception as e:
                        table_counts[table] = f"Error: {e}"
                
                # Check Weaviate health
                try:
                    weaviate_client = get_weaviate_client()
                    weaviate_ready = weaviate_client.is_ready()
                except Exception as e:
                    weaviate_ready = False
                    logger.error(f"Weaviate health check failed: {e}")
                
                return {
                    "database_status": "healthy",
                    "weaviate_status": "healthy" if weaviate_ready else "unhealthy",
                    "table_counts": table_counts,
                    "timestamp": datetime.now().isoformat(),
                    "dependency_manager": "integrated"
                }
        else:
            return {
                "database_status": "unhealthy",
                "error": "Database connection failed",
                "timestamp": datetime.now().isoformat(),
                "dependency_manager": "integrated"
            }
            
    except Exception as e:
        return {
            "database_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "dependency_manager": "error"
        }
