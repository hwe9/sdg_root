import os

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey, DateTime, Table, JSON, Boolean, Index

from sqlalchemy.orm import relationship, sessionmaker, declarative_base

from sqlalchemy.sql import func

from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

Base = declarative_base()


articles_tags = Table(
    'articles_tags',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)


articles_ai_topics = Table(
    'articles_ai_topics',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('ai_topic_id', Integer, ForeignKey('ai_topics.id'))
)

articles_sdg_targets = Table(
    'articles_sdg_targets',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id'), primary_key=True),
    Column('sdg_id', Integer, ForeignKey('sdgs.id'), primary_key=True),
    Column('confidence_score', Float, default=0.0)
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
    name_de = Column(String)  
    name_fr = Column(String)  
    name_es = Column(String)  
    name_zh = Column(String)  
    name_hi = Column(String)
    description = Column(Text)
    description_de = Column(Text) 
    description_fr = Column(Text)
    description_es = Column(Text)
    description_zh = Column(Text)
    description_hi = Column(Text)
    color_hex = Column(String(7))  
    icon_url = Column(String(500)) 
    priority_weight = Column(Float, default=1.0) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    targets = relationship("SDGTarget", back_populates="goal")
    progress = relationship("SdgProgress", back_populates="sdg")
    articles_multi = relationship("Article", secondary=articles_sdg_targets, back_populates="sdgs_multi")
class SDGTarget(Base):
    __tablename__ = "sdg_targets"
    target_id = Column(String(10), primary_key=True)  # "1.1", "1.2", etc.
    goal_id = Column(Integer, ForeignKey("sdgs.id"))
    title_en = Column(Text, nullable=False)
    title_de = Column(Text)
    title_fr = Column(Text)
    title_es = Column(Text)
    title_zh = Column(Text)
    title_hi = Column(Text)

    description = Column(Text)
    description_de = Column(Text)
    description_fr = Column(Text)
    description_es = Column(Text)
    description_zh = Column(Text)
    description_hi = Column(Text)

    target_type = Column(String(50)) 
    deadline_year = Column(Integer, default=2030)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    goal = relationship("Sdg", back_populates="targets")
    indicators = relationship("SDGIndicator", back_populates="target")
class SDGIndicator(Base):
    __tablename__ = "sdg_indicators"
    indicator_id = Column(String(20), primary_key=True)
    target_id = Column(String(10), ForeignKey("sdg_targets.target_id"))
    title_en = Column(Text, nullable=False)
    title_de = Column(Text)
    title_fr = Column(Text)
    title_es = Column(Text)
    title_zh = Column(Text)
    title_hi = Column(Text)
    
    unit_of_measurement = Column(Text)
    data_source = Column(Text)
    methodology = Column(Text)
    tier_classification = Column(String(10))
    custodian_agency = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    target = relationship("SDGTarget", back_populates="indicators")


class Actor(Base):
    __tablename__ = "actors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)
    country_code = Column(String(3), index=True) 
    region = Column(String(100), index=True) 
    is_active = Column(Boolean, default=True) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    progress = relationship("SdgProgress", back_populates="actor")

class AiTopic(Base):
    __tablename__ = "ai_topics"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text) 
    category = Column(String(100), index=True) 
    sdg_relevance = Column(JSON) 
    maturity_level = Column(String(50)) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    articles = relationship("Article", secondary=articles_ai_topics, back_populates="ai_topics")

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content_original = Column(Text)
    content_english = Column(Text)
    summary = Column(Text)
    keywords = Column(Text)
    sdg_id = Column(Integer, ForeignKey("sdgs.id"), index=True)
    sdg_confidence = Column(Float, default=0.0)
    
    authors = Column(Text)
    publication_year = Column(Integer)
    publication_date = Column(DateTime(timezone=True), index=True)
    publisher = Column(String)
    doi = Column(String, unique=True)
    isbn = Column(String, unique=True)
    region = Column(Text(100), index=True)
    country_code = Column(String(3), index=True) 
    language = Column(String(5), default="en", index=True) 
    
    context = Column(String)
    study_type = Column(String, index=True)
    research_methods = Column(String)
    data_sources = Column(String)
    funding = Column(Text)
    funding_info = Column(String)
    bias_indicators = Column(String)
    abstract_original = Column(Text)
    abstract_english = Column(Text)
    relevance_questions = Column(String)
    source_url = Column(String(2000), unique=True, index=True)
    availability = Column(String)
    
    citation_count = Column(Integer, default=0)
    impact_metrics = Column(JSON)
    impact_factor = Column(Float)
    policy_impact = Column(String)

    has_embeddings = Column(Boolean, default=False, index=True)
    embedding_model = Column(String(100))
    embedding_dimension = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    tags = relationship("Tag", secondary=articles_tags, back_populates="articles")
    ai_topics = relationship("AiTopic", secondary=articles_ai_topics, back_populates="articles")
    image_paths = relationship("Image", back_populates="article")
    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")
    sdgs_multi = relationship("Sdg", secondary=articles_sdg_targets, back_populates="articles_multi")
    primary_sdg = relationship("Sdg", foreign_keys=[sdg_id])


class ArticleChunk(Base):
    __tablename__ = "article_chunks"
    id = Column(Integer, primary_key=True, index=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    chunk_id = Column(Integer)
    chunk_order = Column(Integer, index=True)
    text = Column(Text)
    chunk_length = Column(Integer)
    sdg_section = Column(String, index=True)
    sub_section_id = Column(Integer)
    sdg_relevance_scores = Column(JSON) 
    confidence_score = Column(Float, default=0.0)
    has_embedding = Column(Boolean, default=False)
    embedding_hash = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    article = relationship("Article", back_populates="chunks")

class SdgProgress(Base):
    __tablename__ = "sdg_progress"
    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, ForeignKey("actors.id"), index=True)
    sdg_id = Column(Integer, ForeignKey("sdgs.id"), index=True)
    score = Column(Float)
    year = Column(Integer)
    progress_status = Column(String(50)) 
    trend_direction = Column(String(20)) 
    data_quality = Column(String(20), default="medium") 
    data_sources = Column(JSON) 
    confidence_level = Column(Float, default=0.5) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    actor = relationship("Actor", back_populates="progress")
    sdg = relationship("Sdg", back_populates="progress")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    category = Column(String(100), index=True) 
    usage_count = Column(Integer, default=0)
    articles = relationship("Article", secondary=articles_tags, back_populates="articles")


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey('articles.id'), index=True)
    original_path = Column(String(500))
    ocr_text = Column(Text)
    page = Column(Integer)
    caption = Column(Text)
    sdg_tags = Column(JSON)
    ai_tags = Column(Text)
    image_type = Column(String)
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String(10))
    processed_at = Column(DateTime(timezone=True))
    article = relationship("Article", back_populates="image_paths")

class SdgInterlinkage(Base):
    """SDG goal interlinkages"""
    __tablename__ = "sdg_interlinkages"
    
    id = Column(Integer, primary_key=True, index=True)
    from_sdg_id = Column(Integer, ForeignKey("sdgs.id"), index=True)
    to_sdg_id = Column(Integer, ForeignKey("sdgs.id"), index=True)
    relationship_type = Column(String(50)) 
    strength = Column(Float, nullable=False) 
    evidence_level = Column(String(20), default="medium")
    source = Column(String(200))
    
    from_sdg = relationship("Sdg", foreign_keys=[from_sdg_id])
    to_sdg = relationship("Sdg", foreign_keys=[to_sdg_id])


Index('idx_articles_sdg_region_year', Article.sdg_id, Article.region, Article.publication_year)
Index('idx_articles_language_quality', Article.language, Article.content_quality_score)
Index('idx_articles_has_embeddings', Article.has_embeddings)
Index('idx_chunks_article_order', ArticleChunk.article_id, ArticleChunk.chunk_order)
Index('idx_progress_actor_sdg_year', SdgProgress.actor_id, SdgProgress.sdg_id, SdgProgress.year)

Base.metadata.create_all(bind=engine)
