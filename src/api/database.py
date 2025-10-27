# src/api/database.py
from ..core.db_utils import get_database_url
from ..core.db_utils import get_database_engine
from ..core.db_utils import check_database_health
from sqlalchemy.orm import sessionmaker

# Use centralized database configuration
DATABASE_URL = get_database_url()
engine = get_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()