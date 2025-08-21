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
    content_original: Optional[str]
    content_english: Optional[str]
    keywords: Optional[str]
    sdg_id: Optional[int]
    authors: Optional[str]
    publication_year: Optional[int]
    publisher: Optional[str]
    doi: Optional[str]
    isbn: Optional[str]
    region: Optional[str]
    context: Optional[str]
    study_type: Optional[str]
    research_methods: Optional[str]
    data_sources: Optional[str]
    funding: Optional[str]
    funding_info: Optional[str]
    bias_indicators: Optional[str]
    abstract_original: Optional[str]
    abstract_english: Optional[str]
    relevance_questions: Optional[str]
    source_url: Optional[str]
    availability: Optional[str]
    citation_count: Optional[int]
    impact_metrics: Optional[dict]
    impact_factor: Optional[float]
    policy_impact: Optional[str]
    tags: Optional[List[str]] = []
    ai_topics: Optional[List[str]] = []

class ImageBase(BaseModel):
    article_id: int
    original_path: str
    ocr_text: Optional[str]
    page: Optional[int]
    caption: Optional[str]
    sdg_tags: Optional[dict]
    ai_tags: Optional[str]
    image_type: Optional[str]

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int
    created_at: datetime
    tags: List[Tag] = []
    ai_topics: List[AiTopic] = []

    class Config:
        from_attributes = True
