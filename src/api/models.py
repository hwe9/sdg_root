import os

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey, DateTime, Table, JSON, Boolean

from sqlalchemy.orm import relationship, sessionmaker, declarative_base

from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)

Base = declarative_base()

# Many-to-many Join-Tabelle für Artikel und Tags

articles_tags = Table(
    'articles_tags',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

# Many-to-many Join-Tabelle für Artikel und KI-Themen

articles_ai_topics = Table(
    'articles_ai_topics',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('ai_topic_id', Integer, ForeignKey('ai_topics.id'))
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Sdg(Base):
    __tablename__ = "sdgs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    progress = relationship("SdgProgress", back_populates="sdg")

class Actor(Base):
    __tablename__ = "actors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)
    country_code = Column(String)
    progress = relationship("SdgProgress", back_populates="actor")

class AiTopic(Base):
    __tablename__ = "ai_topics"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    articles = relationship("Article", secondary=articles_ai_topics, back_populates="ai_topics")

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content_original = Column(Text)
    content_english = Column(Text)
    keywords = Column(Text)
    sdg_id = Column(Integer, ForeignKey("sdgs.id"))
    authors = Column(Text)
    publication_year = Column(Integer)
    publisher = Column(String)
    doi = Column(String, unique=True)
    isbn = Column(String, unique=True)
    region = Column(Text)
    context = Column(String)
    study_type = Column(String)
    research_methods = Column(String)
    data_sources = Column(String)
    funding = Column(Text)
    funding_info = Column(String)
    bias_indicators = Column(String)
    abstract_original = Column(Text)
    abstract_english = Column(Text)
    relevance_questions = Column(String)
    source_url = Column(String)
    availability = Column(String)
    citation_count = Column(Integer)
    impact_metrics = Column(JSON)
    impact_factor = Column(Float)
    policy_impact = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = relationship("Tag", secondary=articles_tags, back_populates="articles")
    ai_topics = relationship("AiTopic", secondary=articles_ai_topics, back_populates="ai_topics")
    image_paths = relationship("Image", back_populates="article")

class SdgProgress(Base):
    __tablename__ = "sdg_progress"
    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, ForeignKey("actors.id"))
    sdg_id = Column(Integer, ForeignKey("sdgs.id"))
    score = Column(Float)
    year = Column(Integer)
    actor = relationship("Actor", back_populates="progress")
    sdg = relationship("Sdg", back_populates="progress")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    articles = relationship("Article", secondary=articles_tags, back_populates="articles")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey('articles.id'))
    original_path = Column(String)
    ocr_text = Column(Text)
    page = Column(Integer)
    caption = Column(Text)
    sdg_tags = Column(JSON)
    ai_tags = Column(Text)
    image_type = Column(String)
    article = relationship("Article", back_populates="image_paths")

Base.metadata.create_all(bind=engine)
