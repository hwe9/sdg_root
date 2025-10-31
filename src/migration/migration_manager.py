# src/migration/migration_manager.py
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from ..core.db_utils import get_database_url, get_database_engine

logger = logging.getLogger(__name__)

class MigrationManager:
    """Comprehensive database migration management system"""
    
    def __init__(self):
        self.engine = get_database_engine()
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if it doesn't exist"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        rollback_sql TEXT,
                        checksum VARCHAR(64)
                    )
                """))
                conn.commit()
        except SQLAlchemyError as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT migration_name FROM schema_migrations ORDER BY applied_at"))
                return [row[0] for row in result]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def apply_migration(self, migration_name: str, sql_content: str, rollback_sql: str = None) -> bool:
        """Apply a single migration"""
        try:
            with self.engine.begin() as conn:
                # Execute the migration SQL
                conn.execute(text(sql_content))
                
                # Record the migration
                conn.execute(text("""
                    INSERT INTO schema_migrations (migration_name, rollback_sql, checksum)
                    VALUES (:name, :rollback, :checksum)
                """), {
                    "name": migration_name,
                    "rollback": rollback_sql,
                    "checksum": self._calculate_checksum(sql_content)
                })
                
            logger.info(f"Successfully applied migration: {migration_name}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to apply migration {migration_name}: {e}")
            return False
    
    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration"""
        try:
            with self.engine.connect() as conn:
                # Get rollback SQL
                result = conn.execute(text("""
                    SELECT rollback_sql FROM schema_migrations 
                    WHERE migration_name = :name
                """), {"name": migration_name})
                
                rollback_sql = result.fetchone()
                if not rollback_sql or not rollback_sql[0]:
                    logger.error(f"No rollback SQL found for migration: {migration_name}")
                    return False
                
                # Execute rollback
                with self.engine.begin() as conn:
                    conn.execute(text(rollback_sql[0]))
                    
                    # Remove migration record
                    conn.execute(text("""
                        DELETE FROM schema_migrations WHERE migration_name = :name
                    """), {"name": migration_name})
                
            logger.info(f"Successfully rolled back migration: {migration_name}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to rollback migration {migration_name}: {e}")
            return False
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for migration content"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def run_all_pending_migrations(self) -> Dict[str, Any]:
        """Run all pending migrations"""
        results = {
            "applied": [],
            "failed": [],
            "skipped": []
        }
        
        applied_migrations = self.get_applied_migrations()
        
        # Define migrations in order
        migrations = [
            {
                "name": "001_consolidate_users",
                "file": "consolidate_users.sql",
                "rollback_file": "rollback_consolidate_users.sql"
            },
            {
                "name": "002_add_content_hash_to_articles",
                "file": "002_add_content_hash_to_articles.sql",
                "rollback_file": "rollback_add_content_hash_to_articles.sql"
            }
        ]
        
        for migration in migrations:
            if migration["name"] in applied_migrations:
                results["skipped"].append(migration["name"])
                continue
            
            migration_file = self.migrations_dir / migration["file"]
            rollback_file = self.migrations_dir / migration["rollback_file"]
            
            if not migration_file.exists():
                logger.warning(f"Migration file not found: {migration_file}")
                results["failed"].append(migration["name"])
                continue
            
            try:
                sql_content = migration_file.read_text()
                rollback_content = rollback_file.read_text() if rollback_file.exists() else None
                
                if self.apply_migration(migration["name"], sql_content, rollback_content):
                    results["applied"].append(migration["name"])
                else:
                    results["failed"].append(migration["name"])
            except Exception as e:
                logger.error(f"Error processing migration {migration['name']}: {e}")
                results["failed"].append(migration["name"])
        
        return results
    
    def validate_database_schema(self) -> Dict[str, Any]:
        """Validate that database schema is consistent"""
        validation_results = {
            "valid": True,
            "issues": [],
            "tables": {}
        }
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            # Check for required tables
            required_tables = ["users", "sdgs", "articles", "schema_migrations"]
            for table in required_tables:
                if table not in tables:
                    validation_results["valid"] = False
                    validation_results["issues"].append(f"Missing required table: {table}")
                else:
                    validation_results["tables"][table] = "exists"
            
            # Check users table structure
            if "users" in tables:
                columns = inspector.get_columns("users")
                column_names = [col["name"] for col in columns]
                
                required_columns = ["id", "username", "email", "hashed_password", "role", "is_active"]
                for col in required_columns:
                    if col not in column_names:
                        validation_results["valid"] = False
                        validation_results["issues"].append(f"Missing column {col} in users table")
            
            # Check for auth_users table (should not exist after migration)
            if "auth_users" in tables:
                validation_results["valid"] = False
                validation_results["issues"].append("auth_users table still exists - migration may be incomplete")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Schema validation error: {e}")
        
        return validation_results

# Global migration manager instance
migration_manager = MigrationManager()

