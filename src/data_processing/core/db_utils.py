import os
import time
import weaviate
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from typing import Dict, Any, List

DATABASE_URL = os.environ.get("DATABASE_URL")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")

def get_weaviate_client():
    """Erstellt eine Weaviate-Client-Instanz."""
    return weaviate.Client(url=WEAVIATE_URL)

def save_to_database(
    metadata: Dict[str, Any],
    text_content: str,
    embeddings: List[float],
    chunks_data: List[Dict] = None
):
    """
    Speichert Metadaten in PostgreSQL und Vektoren in Weaviate.
    Optional: Speichert Chunks, falls chunks_data übergeben wird.
    """
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            # Save article as usual
            insert_query = text("""
                INSERT INTO articles (
                    title, content_original, content_english, keywords, sdg_id, authors, publication_year,
                    publisher, doi, isbn, region, context, study_type, research_methods, data_sources,
                    funding, funding_info, bias_indicators, abstract_original, abstract_english,
                    relevance_questions, source_url, availability, citation_count, impact_metrics,
                    impact_factor, policy_impact
                ) VALUES (
                    :title, :content_original, :content_english, :keywords, :sdg_id, :authors, :publication_year,
                    :publisher, :doi, :isbn, :region, :context, :study_type, :research_methods, :data_sources,
                    :funding, :funding_info, :bias_indicators, :abstract_original, :abstract_english,
                    :relevance_questions, :source_url, :availability, :citation_count, :impact_metrics,
                    :impact_factor, :policy_impact
                ) RETURNING id
            """)

            result = connection.execute(insert_query, {
                "title": metadata.get('title'),
                "content_original": metadata.get('content_original', text_content),
                "content_english": metadata.get('content_english'),
                "keywords": metadata.get('keywords'),
                "sdg_id": metadata.get('sdg_id'),
                "authors": metadata.get('authors'),
                "publication_year": metadata.get('publication_year'),
                "publisher": metadata.get('publisher'),
                "doi": metadata.get('doi'),
                "isbn": metadata.get('isbn'),
                "region": metadata.get('region'),
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
                "citation_count": metadata.get('citation_count'),
                "impact_metrics": metadata.get('impact_metrics'),
                "impact_factor": metadata.get('impact_factor'),
                "policy_impact": metadata.get('policy_impact')
            })
            article_id = result.scalar_one()

            # --- Tag relations exactly as before ---
            for tag_name in metadata.get('tags', []):
                tag_id = connection.execute(
                    text("SELECT id FROM tags WHERE name = :name"),
                    {"name": tag_name}
                ).scalar_one_or_none()
                if not tag_id:
                    tag_id = connection.execute(
                        text("INSERT INTO tags (name) VALUES (:name) RETURNING id"),
                        {"name": tag_name}
                    ).scalar_one()
                connection.execute(
                    text("INSERT INTO articles_tags (article_id, tag_id) VALUES (:article_id, :tag_id)"),
                    {"article_id": article_id, "tag_id": tag_id}
                )

            # --- AI topics relations exactly as before ---
            for topic_name in metadata.get('ai_topics', []):
                topic_id = connection.execute(
                    text("SELECT id FROM ai_topics WHERE name = :name"),
                    {"name": topic_name}
                ).scalar_one_or_none()
                if not topic_id:
                    topic_id = connection.execute(
                        text("INSERT INTO ai_topics (name) VALUES (:name) RETURNING id"),
                        {"name": topic_name}
                    ).scalar_one()
                connection.execute(
                    text("INSERT INTO articles_ai_topics (article_id, ai_topic_id) VALUES (:article_id, :topic_id)"),
                    {"article_id": article_id, "topic_id": topic_id}
                )

            # --- NEW: Save chunks if provided ---
            if chunks_data:
                for i, chunk_data in enumerate(chunks_data):
                    chunk_query = text("""
                        INSERT INTO article_chunks (
                            article_id, chunk_order, text, chunk_length, sdg_section, confidence_score
                        ) VALUES (
                            :article_id, :chunk_order, :text, :chunk_length, :sdg_section, :confidence_score
                        )
                    """)
                    connection.execute(chunk_query, {
                        "article_id": article_id,
                        "chunk_order": i,
                        "text": chunk_data["text"],
                        "chunk_length": len(chunk_data["text"]),
                        "sdg_section": chunk_data.get("sdg_section", "general"),
                        "confidence_score": chunk_data.get("confidence_score", 0.0)
                    })

            connection.commit()
            print(f"Metadaten für Artikel {article_id} in PostgreSQL gespeichert.")

        # --- Vector: always save in Weaviate, as in original code ---
        client = get_weaviate_client()
        client.schema.get()
        try:
            client.schema.get("ArticleVector")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            class_obj = {
                "class": "ArticleVector",
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "articleId", "dataType": ["int"]}
                ]
            }
            client.schema.create_class(class_obj)

        data_object = {"text": text_content, "articleId": article_id}
        client.data_object.create(data_object=data_object, class_name="ArticleVector", vector=embeddings)
        print(f"Vektor für Artikel {article_id} in Weaviate gespeichert.")

    except OperationalError as e:
        print(f"Datenbankverbindung fehlgeschlagen: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"Fehler beim Speichern in der Datenbank: {e}")


