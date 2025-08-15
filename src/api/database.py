import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Lese die Datenbank-URL aus den Umgebungsvariablen
# Die Umgebungsvariable wird in der docker-compose.yml definiert
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables.")

# Erstelle die Datenbank-Engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Erstelle einen Session-Konstruktor
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Abhängigkeitsfunktion zur Verwaltung der Datenbank-Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()