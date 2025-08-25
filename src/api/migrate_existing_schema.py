"""
Migration script to enhance existing schema without breaking changes
"""
import logging
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, String, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def migrate_existing_schema():

    logger.info("Starting schema migration...")
    
    with engine.begin() as conn:
        
        try:
            conn.execute(text("""
                ALTER TABLE sdgs 
                ADD COLUMN IF NOT EXISTS goal_number INTEGER,
                ADD COLUMN IF NOT EXISTS name_de VARCHAR,
                ADD COLUMN IF NOT EXISTS name_fr VARCHAR,
                ADD COLUMN IF NOT EXISTS name_es VARCHAR,
                ADD COLUMN IF NOT EXISTS name_zh VARCHAR,
                ADD COLUMN IF NOT EXISTS name_hi VARCHAR,
                ADD COLUMN IF NOT EXISTS description_de TEXT,
                ADD COLUMN IF NOT EXISTS description_fr TEXT,
                ADD COLUMN IF NOT EXISTS description_es TEXT,
                ADD COLUMN IF NOT EXISTS description_zh TEXT,
                ADD COLUMN IF NOT EXISTS description_hi TEXT,
                ADD COLUMN IF NOT EXISTS color_hex VARCHAR(7),
                ADD COLUMN IF NOT EXISTS icon_url VARCHAR(500),
                ADD COLUMN IF NOT EXISTS priority_weight FLOAT DEFAULT 1.0,
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE
            """))
            logger.info("✅ Enhanced SDGs table")
        except Exception as e:
            logger.warning(f"SDGs table enhancement: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE articles 
                ADD COLUMN IF NOT EXISTS summary TEXT,
                ADD COLUMN IF NOT EXISTS sdg_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS publication_date TIMESTAMP WITH TIME ZONE,
                ADD COLUMN IF NOT EXISTS country_code VARCHAR(3),
                ADD COLUMN IF NOT EXISTS language VARCHAR(5) DEFAULT 'en',
                ADD COLUMN IF NOT EXISTS word_count INTEGER,
                ADD COLUMN IF NOT EXISTS readability_score FLOAT,
                ADD COLUMN IF NOT EXISTS content_quality_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS has_embeddings BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
                ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER,
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE,
                ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP WITH TIME ZONE
            """))
            logger.info("✅ Enhanced Articles table")
        except Exception as e:
            logger.warning(f"Articles table enhancement: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE article_chunks 
                ADD COLUMN IF NOT EXISTS chunk_order INTEGER,
                ADD COLUMN IF NOT EXISTS sdg_relevance_scores JSON,
                ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS has_embedding BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS embedding_hash VARCHAR(64)
            """))
            logger.info("✅ Enhanced ArticleChunks table")
        except Exception as e:
            logger.warning(f"ArticleChunks table enhancement: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE actors 
                ADD COLUMN IF NOT EXISTS region VARCHAR(100),
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """))
            logger.info("✅ Enhanced Actors table")
        except Exception as e:
            logger.warning(f"Actors table enhancement: {e}")

        
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sdg_interlinkages (
                    id SERIAL PRIMARY KEY,
                    from_sdg_id INTEGER REFERENCES sdgs(id),
                    to_sdg_id INTEGER REFERENCES sdgs(id),
                    relationship_type VARCHAR(50) NOT NULL,
                    strength FLOAT NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
                    evidence_level VARCHAR(20) DEFAULT 'medium',
                    source VARCHAR(200),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            logger.info("✅ Created SDG Interlinkages table")
        except Exception as e:
            logger.warning(f"SDG Interlinkages table creation: {e}")

        
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS articles_sdg_targets (
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                    sdg_id INTEGER REFERENCES sdgs(id) ON DELETE CASCADE,
                    confidence_score FLOAT DEFAULT 0.0,
                    PRIMARY KEY (article_id, sdg_id)
                )
            """))
            logger.info("✅ Created Articles-SDGs many-to-many table")
        except Exception as e:
            logger.warning(f"Articles-SDGs table creation: {e}")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_articles_sdg_id ON articles(sdg_id)",
            "CREATE INDEX IF NOT EXISTS idx_articles_region ON articles(region)",
            "CREATE INDEX IF NOT EXISTS idx_articles_publication_year ON articles(publication_year)",
            "CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language)",
            "CREATE INDEX IF NOT EXISTS idx_articles_has_embeddings ON articles(has_embeddings)",
            "CREATE INDEX IF NOT EXISTS idx_sdg_progress_year ON sdg_progress(year)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_article_order ON article_chunks(article_id, chunk_order)",
            "CREATE INDEX IF NOT EXISTS idx_sdgs_goal_number ON sdgs(goal_number)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Index creation: {e}")
        
        logger.info("✅ Created performance indexes")

        try:
            sdg_updates = [
                ("No Poverty", 1, "#E5243B"),
                ("Zero Hunger", 2, "#DDA63A"),
                ("Good Health and Well-being", 3, "#4C9F38"),
                ("Quality Education", 4, "#C5192D"),
                ("Gender Equality", 5, "#FF3A21"),
                ("Clean Water and Sanitation", 6, "#26BDE2"),
                ("Affordable and Clean Energy", 7, "#FCC30B"),
                ("Decent Work and Economic Growth", 8, "#A21942"),
                ("Industry, Innovation and Infrastructure", 9, "#FD6925"),
                ("Reduced Inequalities", 10, "#DD1367"),
                ("Sustainable Cities and Communities", 11, "#FD9D24"),
                ("Responsible Consumption and Production", 12, "#BF8B2E"),
                ("Climate Action", 13, "#3F7E44"),
                ("Life Below Water", 14, "#0A97D9"),
                ("Life on Land", 15, "#56C02B"),
                ("Peace, Justice and Strong Institutions", 16, "#00689D"),
                ("Partnerships for the Goals", 17, "#19486A")
            ]
            
            for name, goal_number, color in sdg_updates:
                conn.execute(text("""
                    UPDATE sdgs 
                    SET goal_number = :goal_number, color_hex = :color
                    WHERE name ILIKE :name AND goal_number IS NULL
                """), {"name": f"%{name}%", "goal_number": goal_number, "color": color})
            
            logger.info("✅ Updated existing SDG data")
        except Exception as e:
            logger.warning(f"SDG data update: {e}")
        
            conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sdg_targets (
                        target_id VARCHAR(10) PRIMARY KEY,
                        goal_id INTEGER REFERENCES sdgs(id) ON DELETE CASCADE,
                        title_en TEXT NOT NULL,
                        title_de TEXT,
                        title_fr TEXT,
                        title_es TEXT,
                        title_zh TEXT,
                        title_hi TEXT,
                        description TEXT,
                        description_de TEXT,
                        description_fr TEXT,
                        description_es TEXT,
                        description_zh TEXT,
                        description_hi TEXT,
                        target_type VARCHAR(50),
                        deadline_year INTEGER DEFAULT 2030,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE
                    );
                """))
                

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sdg_indicators (
                    indicator_id VARCHAR(20) PRIMARY KEY,
                    target_id VARCHAR(10) REFERENCES sdg_targets(target_id) ON DELETE CASCADE,
                    title_en TEXT NOT NULL,
                    title_de TEXT,
                    title_fr TEXT,
                    title_es TEXT,
                    title_zh TEXT,
                    title_hi TEXT,
                    unit_of_measurement TEXT,
                    data_source TEXT,
                    methodology TEXT,
                    tier_classification VARCHAR(10),
                    custodian_agency TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """))
            
            # Create indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sdg_targets_goal_id ON sdg_targets(goal_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sdg_indicators_target_id ON sdg_indicators(target_id);"))
                
            logger.info("✅ Created SDG targets and indicators tables")
            
        except Exception as e:
            logger.error(f"Error creating SDG targets/indicators: {e}")

    logger.info("Schema migration completed successfully!")



if __name__ == "__main__":
    migrate_existing_schema()
