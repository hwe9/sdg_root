# src/core/models.py
import os
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import func
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from ..core.db_utils import get_database_url
from ..core.db_utils import get_database_engine

# Use centralized database configuration
DATABASE_URL = get_database_url()
engine = get_database_engine()
Base = declarative_base()

class User(Base):
    """Unified User model for all services"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    email = Column(String(320), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="user", index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    
    __table_args__ = (
        UniqueConstraint("username"), 
        UniqueConstraint("email"),
    )

def init_db():
    Base.metadata.create_all(bind=engine)

