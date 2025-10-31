# src/data_processing/core/db_utils.py

import os
import json
import logging
import time
from contextlib import contextmanager
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
from typing import Callable
import httpx
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import DisconnectionError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from datetime import datetime
from datetime import timedelta

from src.core.metrics import get_histogram
from src.core.metrics import get_counter
from ...core.dependency_manager import get_dependency_manager
from ...core.secrets_manager import secrets_manager  # falls künftig genutzt
from ...core.db_utils import get_database_url
from ...core.db_utils import get_database_engine
from ...core.db_utils import check_database_health

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HEALTH_CHECK_DURATION = get_histogram(
    "health_check_duration_seconds",
    "Duration of health checks",
    labelnames=("target",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HEALTH_CHECK_FAILS = get_counter(
    "health_check_fail_total",
    "Total number of failed health checks",
    labelnames=("target",),
)

# Use centralized database configuration
DATABASE_URL = get_database_url()
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://weaviate_service:8080")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")

def get_database_engine():
    """Use centralized database engine configuration"""
    from ...core.db_utils import get_database_engine as get_centralized_engine
    return get_centralized_engine()

def check_database_health(
    get_engine: Optional[Callable[[], Engine]] = None,
    *,
    target_label: str = "database",
) -> bool:
    """Use centralized database health check"""
    from ...core.db_utils import check_database_health as centralized_health_check
    return centralized_health_check()
           

def get_database_session():
    """Kompatible Wrapper-API wie im Original: Session via Dependency Manager beziehen."""
    dep_manager = get_dependency_manager()
    return dep_manager.get_database_session()

@contextmanager
def get_db_connection():
    engine = get_database_engine()
    for attempt in range(3):
        try:
            with engine.begin() as connection:
                yield connection
                return
        except (OperationalError, DisconnectionError) as e:
            if attempt < 2:
                logger.warning(f"Database operation attempt {attempt+1} failed: {e}")
                time.sleep(1)
            else:
                logger.error("Database operation failed after retries")
                raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise

def _probe(url: str, timeout: float, expect_json: bool) -> httpx.Response:
    headers = {"Accept": "application/json"} if expect_json else None
    return httpx.get(url, timeout=timeout, headers=headers)

def _check_weaviate_health(base_url: str, timeout: float = 5.0) -> bool:
    """Akzeptiert 200 und 204 für /.well-known/ready bzw. /v1/.well-known/ready; 200 für /v1/meta."""
    base = base_url.rstrip("/")
    endpoints = ["/.well-known/ready", "/v1/.well-known/ready", "/v1/meta"]
    for path in endpoints:
        url = f"{base}{path}"
        try:
            resp = _probe(url, timeout, path.endswith("/meta"))
            if resp.status_code in (200, 204):
                return True
        except Exception:
            continue
    return False

def get_weaviate_client():
    import weaviate
    if not _check_weaviate_health(WEAVIATE_URL):
        raise ConnectionError(
            f"Weaviate not ready at {WEAVIATE_URL}; readiness check failed (/.well-known/ready or /v1/meta)"
        )
    auth = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth)
    try:
        client.schema.get("ArticleVector")
    except Exception:
        _create_weaviate_schema(client)
    return client

def _ensure_content_hash_in_schema(client):
    """Ensure content_hash property exists in ArticleVector schema"""
    try:
        # Handle adapter pattern: access underlying Weaviate client
        weaviate_client = getattr(client, '_client', client)
        
        class_obj = weaviate_client.schema.get("ArticleVector")
        existing_properties = {prop.get("name") for prop in class_obj.get("properties", [])}
        
        if "contentHash" not in existing_properties:
            # Add content_hash property to existing schema
            property_definition = {
                "dataType": ["string"],
                "description": "MD5 hash of content for deduplication",
                "name": "contentHash"
            }
            weaviate_client.schema.property.create("ArticleVector", property_definition)
            logger.info("✅ Added contentHash property to ArticleVector schema")
        else:
            logger.debug("contentHash property already exists in ArticleVector schema")
    except Exception as e:
        logger.warning(f"Could not ensure contentHash in schema (may not exist yet): {e}")

def _create_weaviate_schema(client):
    class_obj = {
        "class": "ArticleVector",
        "description": "SDG article embeddings for semantic search",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"], "description": "Article text content"},
            {"name": "articleId", "dataType": ["int"], "description": "Reference to article ID in PostgreSQL"},
            {"name": "chunkId", "dataType": ["int"], "description": "Chunk ID for long documents"},
            {"name": "sdgGoals", "dataType": ["int[]"], "description": "Related SDG goals"},
            {"name": "language", "dataType": ["text"], "description": "Content language"},
            {"name": "region", "dataType": ["text"], "description": "Geographic region"},
            {"name": "contentHash", "dataType": ["string"], "description": "MD5 hash of content for deduplication"},
        ],
    }
    client.schema.create_class(class_obj)
    logger.info("Created ArticleVector schema in Weaviate")
    
    # Ensure content_hash is in schema (in case schema already existed)
    _ensure_content_hash_in_schema(client)

async def save_to_database(metadata: Dict[str, Any], text_content: str, embeddings: List[float], chunks_data: Optional[List[Dict]] = None):
    """Async‑Pfad via Dependency‑Manager wie im Original."""
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
    try:
        # Ensure content_hash property exists in schema
        _ensure_content_hash_in_schema(vector_client)
        
        content_hash = (metadata or {}).get('content_hash', '')
        
        if chunks_data and len(chunks_data) > 0:
            for i, chunk_data in enumerate(chunks_data):
                chunk_vector = chunk_data.get("embedding") or embeddings
                vector_data = {
                    "text": chunk_data["text"],
                    "articleId": article_id,
                    "chunkId": i,
                    "sdgGoals": (metadata or {}).get('sdg_goals', []),
                    "language": (metadata or {}).get('language', 'en'),
                    "region": (metadata or {}).get('region', ''),
                    "contentHash": content_hash
                }
                vector_client.insert_data(vector_data, "ArticleVector", chunk_vector)
        else:
            vector_data = {
                "text": text_content,
                "articleId": article_id,
                "chunkId": 0,
                "sdgGoals": (metadata or {}).get('sdg_goals', []),
                "language": (metadata or {}).get('language', 'en'),
                "region": (metadata or {}).get('region', ''),
                "contentHash": content_hash
            }
            vector_client.insert_data(vector_data, "ArticleVector", embeddings)
        logger.info(f"✅ Vector data for article {article_id} saved via dependency manager")
    except Exception as e:
        logger.error(f"❌ Error saving to Weaviate via client: {e}")
        raise

def _save_to_weaviate(article_id: int, text_content: str, embeddings: List[float],
                      chunks_data: Optional[List[Dict]] = None, metadata: Dict[str, Any] = None):
    """Sync‑Pfad für batch_save_to_database, unveränderte Signatur beibehalten."""
    import weaviate
    client = get_weaviate_client()
    try:
        try:
            client.schema.get("ArticleVector")
            # Ensure content_hash exists in existing schema
            _ensure_content_hash_in_schema(client)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            _create_weaviate_schema(client)
        
        content_hash = (metadata or {}).get('content_hash', '')
        
        if chunks_data and len(chunks_data) > 0:
            for i, chunk_data in enumerate(chunks_data):
                chunk_vector = chunk_data.get("embedding") or embeddings
                vector_data = {
                    "text": chunk_data["text"],
                    "articleId": article_id,
                    "chunkId": i,
                    "sdgGoals": (metadata or {}).get('sdg_goals', []),
                    "language": (metadata or {}).get('language', 'en'),
                    "region": (metadata or {}).get('region', ''),
                    "contentHash": content_hash
                }
                client.data_object.create(
                    data_object=vector_data,
                    class_name="ArticleVector",
                    vector=chunk_vector
                )
        else:
            vector_data = {
                "text": text_content,
                "articleId": article_id,
                "chunkId": 0,
                "sdgGoals": (metadata or {}).get('sdg_goals', []),
                "language": (metadata or {}).get('language', 'en'),
                "region": (metadata or {}).get('region', ''),
                "contentHash": content_hash
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
            word_count, content_quality_score, content_hash, created_at
        ) VALUES (
            :title, :content_original, :content_english, :summary, :keywords, :sdg_id,
            :authors, :publication_year, :publication_date, :publisher, :doi, :isbn,
            :region, :country_code, :language, :context, :study_type, :research_methods,
            :data_sources, :funding, :funding_info, :bias_indicators, :abstract_original,
            :abstract_english, :relevance_questions, :source_url, :availability,
            :citation_count, :impact_metrics, :impact_factor, :policy_impact,
            :word_count, :content_quality_score, :content_hash, NOW()
        ) RETURNING id
    """)
    content_quality_score = _calculate_content_quality_score(metadata, text_content)
    params = {
        "title": (metadata.get('title') or 'Untitled')[:500],
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
        "content_quality_score": content_quality_score,
        "content_hash": metadata.get('content_hash')  # MD5 hash from content validation
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
                text("INSERT INTO tags (name, category, usage_count) VALUES (:name, :category, 1) RETURNING id"),
                {"name": tag_name, "category": "general"}
            ).scalar_one()
        else:
            connection.execute(
                text("UPDATE tags SET usage_count = usage_count + 1 WHERE id = :tag_id"),
                {"tag_id": tag_id}
            )
        try:
            connection.execute(
                text("INSERT INTO articles_tags (article_id, tag_id) VALUES (:article_id, :tag_id) ON CONFLICT (article_id, tag_id) DO NOTHING"),
                {"article_id": article_id, "tag_id": tag_id}
            )
        except IntegrityError:
            pass

def _insert_ai_topic_relationships(connection, article_id: int, metadata: Dict[str, Any]):
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
                text("INSERT INTO ai_topics (name, category, created_at) VALUES (:name, :category, NOW()) RETURNING id"),
                {"name": topic_name, "category": "general"}
            ).scalar_one()
        try:
            connection.execute(
                text("INSERT INTO articles_ai_topics (article_id, ai_topic_id) VALUES (:article_id, :topic_id) ON CONFLICT (article_id, ai_topic_id) DO NOTHING"),
                {"article_id": article_id, "topic_id": topic_id}
            )
        except IntegrityError:
            pass

def _insert_sdg_target_relationships(connection, article_id: int, metadata: Dict[str, Any]):
    sdg_ids: List[int] = []
    if metadata.get('sdg_id'):
        sdg_ids.append(metadata['sdg_id'])
    if metadata.get('sdg_goals'):
        if isinstance(metadata['sdg_goals'], list):
            sdg_ids.extend(metadata['sdg_goals'])
        elif isinstance(metadata['sdg_goals'], int):
            sdg_ids.append(metadata['sdg_goals'])
    sdg_ids = list(set([g for g in sdg_ids if isinstance(g, int) and 1 <= g <= 17]))
    for sdg_id in sdg_ids:
        confidence_score = metadata.get('sdg_confidence', 0.8)
        connection.execute(
            text("""
                INSERT INTO articles_sdg_targets (article_id, sdg_id, confidence_score) 
                VALUES (:article_id, :sdg_id, :confidence_score) 
                ON CONFLICT (article_id, sdg_id) DO UPDATE SET confidence_score = :confidence_score
            """),
            {"article_id": article_id, "sdg_id": sdg_id, "confidence_score": confidence_score}
        )

def _extract_publication_year(metadata: Dict[str, Any]) -> Optional[int]:
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
                    if 1900 <= year_value <= 2030:
                        return year_value
            except (ValueError, TypeError):
                continue
    return None

def _parse_publication_date(date_string: str) -> Optional[datetime]:
    if not date_string:
        return None
    try:
        date_formats = [
            '%Y-%m-%d', '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
            '%Y/%m/%d', '%d/%m/%Y', '%Y'
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_string), fmt)
            except ValueError:
                continue
        import re
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
    title = metadata.get('title', '')
    if title and len(title.strip()) > 10:
        score += 0.15
    elif title and len(title.strip()) > 5:
        score += 0.08
    content_length = len(text_content.strip())
    if content_length > 5000:
        score += 0.25
    elif content_length > 2000:
        score += 0.20
    elif content_length > 500:
        score += 0.10
    elif content_length > 100:
        score += 0.05
    if metadata.get('authors'):
        score += 0.10
    if metadata.get('publication_year') or metadata.get('publisher'):
        score += 0.10
    if metadata.get('doi') or metadata.get('isbn'):
        score += 0.10
    if metadata.get('abstract_original') or metadata.get('abstract_english'):
        score += 0.10
    if metadata.get('source_url'):
        score += 0.05
    if metadata.get('keywords') or metadata.get('tags'):
        score += 0.05
    if metadata.get('sdg_id') or metadata.get('sdg_goals'):
        score += 0.10
    return min(score, 1.0)

def get_article_by_id(article_id: int) -> Optional[Dict[str, Any]]:
    engine = get_database_engine()
    try:
        with engine.connect() as connection:
            article_query = text("""
                SELECT a.*, s.name as sdg_name
                FROM articles a
                LEFT JOIN sdgs s ON a.sdg_id = s.goal_number
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
            article["tags"] = [row._mapping["name"] for row in tags]
            return article
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        return None

def search_similar_content(query_embedding: List[float], limit: int = 10, sdg_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Bewahrt Original-API, nutzt Weaviate-Client direkt."""
    try:
        import weaviate
        client = get_weaviate_client()
        query_builder = (
            client.query
            .get("ArticleVector", ["text", "articleId", "chunkId", "sdgGoals", "language", "region", "contentHash"])
            .with_near_vector({"vector": query_embedding, "certainty": 0.7})
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

def batch_save_to_database(items: List[Dict[str, Any]], batch_size: int = 50):
    """Bewahrt die synchrone Batch-API und nutzt _save_to_weaviate wie im Original."""
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
    """Bewahrt die Cleanup-API; löscht alte Artikel per Retention-Policy."""
    engine = get_database_engine()
    try:
        with engine.begin() as connection:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
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
    """Beibehaltener Statistik-Endpunkt, Join auf goal_number korrigiert."""
    engine = get_database_engine()
    try:
        with engine.connect() as connection:
            stats: Dict[str, Any] = {}
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
                LEFT JOIN articles a ON s.goal_number = a.sdg_id
                GROUP BY s.goal_number, s.name
                ORDER BY article_count DESC
            """)).fetchall()
            stats["sdg_distribution"] = [dict(row._mapping) for row in sdg_stats]
            return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {}

def get_database_health() -> Dict[str, Any]:
    """Beibehaltener Health-Endpoint mit Weaviate-Status."""
    try:
        if check_database_health():
            engine = get_database_engine()
            with engine.connect() as connection:
                tables = ['articles', 'sdgs', 'actors', 'tags', 'ai_topics', 'article_chunks']
                table_counts: Dict[str, Any] = {}
                for table in tables:
                    try:
                        result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = result.scalar()
                    except Exception as e:
                        table_counts[table] = f"Error: {e}"
                try:
                    _ = get_weaviate_client()
                    weaviate_ready = True
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
