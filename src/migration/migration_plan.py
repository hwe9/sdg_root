# src/migration/migration_plan.py
"""
Migration plan for SDG database schema
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_utils import get_database_url, get_database_engine
from core.logging_config import get_logger

logger = get_logger("migration")

def main():
    """Main migration plan execution"""
    logger.info("Starting migration plan execution")
    
    try:
        # Get database connection
        database_url = get_database_url()
        engine = get_database_engine()
        
        logger.info(f"Connected to database: {database_url.split('@')[-1] if '@' in database_url else 'local'}")
        
        # Check if migrations table exists
        with engine.connect() as conn:
            result = conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'migrations'
                );
            """)
            
            migrations_table_exists = result.scalar()
            
            if not migrations_table_exists:
                logger.info("Creating migrations table")
                conn.execute("""
                    CREATE TABLE migrations (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64)
                    );
                """)
                conn.commit()
                logger.info("Migrations table created")
            else:
                logger.info("Migrations table already exists")
        
        logger.info("Migration plan completed successfully")
        
    except Exception as e:
        logger.error(f"Migration plan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
