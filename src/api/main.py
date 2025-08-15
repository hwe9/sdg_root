from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import models, schemas
from database import get_db

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API Service is running!"}

# --- CRUD Endpunkte für Articles ---

@app.post("/articles/", response_model=schemas.Article)
def create_article(article: schemas.ArticleCreate, db: Session = Depends(get_db)):
    db_article = models.Article(**article.dict())
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

@app.get("/articles/", response_model=List[schemas.Article])
def read_articles(db: Session = Depends(get_db)):
    articles = db.query(models.Article).all()
    return articles

@app.get("/articles/{article_id}", response_model=schemas.Article)
def read_article(article_id: int, db: Session = Depends(get_db)):
    db_article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if db_article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return db_article

# --- CRUD Endpunkte für SDGs ---

@app.post("/sdgs/", response_model=schemas.Sdg)
def create_sdg(sdg: schemas.SdgCreate, db: Session = Depends(get_db)):
    db_sdg = models.Sdg(**sdg.dict())
    db.add(db_sdg)
    db.commit()
    db.refresh(db_sdg)
    return db_sdg

@app.get("/sdgs/", response_model=List[schemas.Sdg])
def read_sdgs(db: Session = Depends(get_db)):
    sdgs = db.query(models.Sdg).all()
    return sdgs

# --- CRUD Endpunkte für Actors ---

@app.post("/actors/", response_model=schemas.Actor)
def create_actor(actor: schemas.ActorCreate, db: Session = Depends(get_db)):
    db_actor = models.Actor(**actor.dict())
    db.add(db_actor)
    db.commit()
    db.refresh(db_actor)
    return db_actor

@app.get("/actors/", response_model=List[schemas.Actor])
def read_actors(db: Session = Depends(get_db)):
    actors = db.query(models.Actor).all()
    return actors