from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class AiTopicBase(BaseModel):
    name: str

class AiTopicCreate(AiTopicBase):
    pass

class AiTopic(AiTopicBase):
    id: int
    class Config:
        from_attributes = True

class TagBase(BaseModel):
    name: str

class TagCreate(TagBase):
    pass

class Tag(TagBase):
    id: int
    class Config:
        from_attributes = True

class SdgBase(BaseModel):
    name: str
    description: str

class SdgCreate(SdgBase):
    pass

class Sdg(SdgBase):
    id: int
    class Config:
        from_attributes = True

class ActorBase(BaseModel):
    name: str
    type: str
    country_code: Optional[str] = None

class ActorCreate(ActorBase):
    pass

class Actor(ActorBase):
    id: int
    class Config:
        from_attributes = True

class SdgProgressBase(BaseModel):
    actor_id: int
    sdg_id: int
    score: float
    year: int

class SdgProgress(SdgProgressBase):
    id: int
    class Config:
        from_attributes = True

class ArticleBase(BaseModel):
    title: str
    content: str
    sdg_id: Optional[int] = None
    
    # Bibliographische Daten
    authors: Optional[str] = None
    publication_year: Optional[int] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    # Regionale Relevanz
    region: Optional[str] = None
    context: Optional[str] = None
    
    # Forschungsdesign und Methode
    study_type: Optional[str] = None
    research_methods: Optional[str] = None
    data_sources: Optional[str] = None
    
    # Finanzierer und Interessenlage
    funding_info: Optional[str] = None
    bias_indicators: Optional[str] = None
    
    # Kurzbeschreibung und Abstract
    abstract: Optional[str] = None
    relevance_questions: Optional[str] = None
    
    # Zitierlink/Weblink
    source_url: Optional[str] = None
    availability: Optional[str] = None
    
    # Evaluations- und Impact-Informationen
    citation_count: Optional[int] = None
    impact_factor: Optional[float] = None
    policy_impact: Optional[str] = None

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int
    created_at: datetime
    tags: List[Tag] = []
    ai_topics: List[AiTopic] = []
    class Config:
        from_attributes = True