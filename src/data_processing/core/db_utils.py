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

def save_to_database(metadata: Dict[str, Any], text_content: str, embeddings: List[float]):
    """
    Speichert Metadaten in PostgreSQL und Vektoren in Weaviate.
    """
    try:
        # PostgreSQL-Teil
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            # Erweitert, um alle neuen Felder zu berücksichtigen
            insert_query = text("""
                INSERT INTO articles (
                    title, content, source_url, authors, publication_year, 
                    publisher, doi, isbn, region, context, study_type, 
                    research_methods, data_sources, funding_info, 
                    bias_indicators, abstract, relevance_questions, 
                    availability, citation_count, impact_factor, policy_impact
                ) VALUES (
                    :title, :content, :source_url, :authors, :publication_year, 
                    :publisher, :doi, :isbn, :region, :context, :study_type, 
                    :research_methods, :data_sources, :funding_info, 
                    :bias_indicators, :abstract, :relevance_questions, 
                    :availability, :citation_count, :impact_factor, :policy_impact
                ) RETURNING id
            """)
            result = connection.execute(insert_query, {
                "title": metadata.get('title', 'Untitled'),
                "content": text_content,
                "source_url": metadata.get('source_url', 'Unknown'),
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
                "funding_info": metadata.get('funding_info'),
                "bias_indicators": metadata.get('bias_indicators'),
                "abstract": metadata.get('abstract'),
                "relevance_questions": metadata.get('relevance_questions'),
                "availability": metadata.get('availability'),
                "citation_count": metadata.get('citation_count'),
                "impact_factor": metadata.get('impact_factor'),
                "policy_impact": metadata.get('policy_impact')
            })
            article_id = result.scalar_one()
            
            for tag_name in metadata['tags']:
                tag_id = connection.execute(text("SELECT id FROM tags WHERE name = :name"), {"name": tag_name}).scalar_one_or_none()
                if not tag_id:
                    tag_id = connection.execute(text("INSERT INTO tags (name) VALUES (:name) RETURNING id"), {"name": tag_name}).scalar_one()
                connection.execute(text("INSERT INTO articles_tags (article_id, tag_id) VALUES (:article_id, :tag_id)"), {"article_id": article_id, "tag_id": tag_id})

            connection.commit()
            print(f"Metadaten für Artikel {article_id} in PostgreSQL gespeichert.")

        # Weaviate-Teil
        client = get_weaviate_client()
        client.schema.get() # Verbindung prüfen
        
        # Erstelle eine Schema-Klasse in Weaviate, falls sie nicht existiert
        try:
            client.schema.get("ArticleVector")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            class_obj = {
                "class": "ArticleVector",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "articleId", "dataType": ["int"]}
                ]
            }
            client.schema.create_class(class_obj)
        
        # Füge den Vektor und die Metadaten hinzu
        data_object = {"text": text_content, "articleId": article_id}
        client.data_object.create(data_object=data_object, class_name="ArticleVector", vector=embeddings)
        print(f"Vektor für Artikel {article_id} in Weaviate gespeichert.")
        
    except OperationalError as e:
        print(f"Datenbankverbindung fehlgeschlagen: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"Fehler beim Speichern in der Datenbank: {e}")