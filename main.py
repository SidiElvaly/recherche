#!/usr/bin/env python3
import os, json, asyncio, logging, traceback
from datetime import date
from math import exp
from typing import List, Optional, Union

import numpy as np, faiss
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from elasticsearch import Elasticsearch, BadRequestError, NotFoundError
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_NAME = "dossiers_medicaux"
EMBEDDING_DIM = 384
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/faiss/faiss_hnsw.index")
FAISS_IDS_PATH = os.getenv("FAISS_IDS_PATH", "/faiss/faiss_ids.json")
ALLOWED_ORIGINS = ["*"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_search")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)

def verify(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if creds is None or creds.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="unauthorized")
    return True

es = Elasticsearch(os.getenv("ES_HOST", "http://elasticsearch:9200"))

model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
except Exception:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

faiss_index: faiss.Index = None
faiss_doc_ids: List[str] = []
faiss_lock = asyncio.Lock()

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

class ConsultationEntry(BaseModel):
    date: Optional[Union[date, str]] = None
    raison: Optional[str] = None
    diagnostic: Optional[str] = None
    traitement: Optional[str] = None
    medecin: Optional[str] = None

class ExamenEntry(BaseModel):
    date: Optional[Union[date, str]] = None
    type_examen: str
    resultat: Optional[str] = None
    unite: Optional[str] = None

class DocumentMedical(BaseModel):
    document_id: str
    consultations: List[ConsultationEntry] = []
    examens: List[ExamenEntry] = []
    allergies_connues: List[str] = []
    traitements_actuels: List[str] = []
    texte_brut: Optional[str] = None
    embedding: Optional[List[float]] = None

class SearchRequest(BaseModel):
    query: str
    text_size: int = 50
    vector_size: int = 50
    final_k: int = 5
    weight_text: float = 1.0
    weight_vector: float = 1.0
    detailed: bool = Query(False)

class SearchResult(BaseModel):
    id: str
    record: DocumentMedical
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    note: Optional[str] = None

def create_index() -> None:
    mapping = {
        "settings": {
            "analysis": {
                "filter": {
                    "synonym_filter": {
                        "type": "synonym",
                        "synonyms": [
                            "myocardial infarction, heart attack",
                            "cancer, malignant tumor"
                        ]
                    }
                },
                "analyzer": {
                    "synonym_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "synonym_filter"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "texte_brut": {"type": "text", "analyzer": "synonym_analyzer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    exists = False
    try:
        exists = es.indices.exists(index=INDEX_NAME)
    except BadRequestError:
        try:
            es.indices.get(index=INDEX_NAME)
            exists = True
        except NotFoundError:
            exists = False
    if not exists:
        es.indices.create(index=INDEX_NAME, body=mapping)
        logger.info("Index %s created", INDEX_NAME)

def load_faiss() -> None:
    global faiss_index, faiss_doc_ids
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_IDS_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        faiss_doc_ids = json.load(open(FAISS_IDS_PATH))
        logger.info("FAISS loaded (%d vectors)", len(faiss_doc_ids))
    else:
        faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_doc_ids = []

def save_faiss() -> None:
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    json.dump(faiss_doc_ids, open(FAISS_IDS_PATH, "w"))

@app.on_event("startup")
async def on_startup():
    create_index()
    load_faiss()

@app.post("/add_document", dependencies=[Depends(verify)])
async def add_document(docs: List[DocumentMedical]) -> dict:
    actions = []
    async with faiss_lock:
        for doc in docs:
            text = " ".join([
                doc.texte_brut or "",
                " ".join(c.raison or "" for c in doc.consultations),
                " ".join(c.diagnostic or "" for c in doc.consultations),
                " ".join(e.type_examen for e in doc.examens),
                " ".join(doc.allergies_connues or []),
                " ".join(doc.traitements_actuels or [])
            ]).strip()
            emb = np.array(model.encode(text), dtype=np.float32)
            emb /= np.linalg.norm(emb)
            doc.embedding = emb.tolist()
            actions.append({"_index": INDEX_NAME, "_id": doc.document_id, "_source": doc.dict()})
            faiss_index.add(emb.reshape(1, -1))
            faiss_doc_ids.append(doc.document_id)
        bulk(es, actions)
        save_faiss()
    return {"inserted": len(docs)}

@app.post("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest) -> SearchResponse:
    try:
        q_emb = np.array(model.encode(req.query), dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb)
        es_query = {
            "size": req.text_size,
            "query": {
                "multi_match": {
                    "query": req.query,
                    "fields": [
                        "texte_brut",
                        "consultations.raison",
                        "consultations.diagnostic",
                        "examens.type_examen",
                        "allergies_connues",
                        "traitements_actuels"
                    ],
                    "analyzer": "synonym_analyzer",
                    "fuzziness": "AUTO",
                    "boost": req.weight_text
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": q_emb.tolist(),
                "k": req.vector_size,
                "num_candidates": req.vector_size * 2,
                "boost": req.weight_vector
            }
        }
        es_hits = es.search(index=INDEX_NAME, body=es_query)["hits"]["hits"]
        es_scores = {h["_id"]: h["_score"] for h in es_hits}
        async with faiss_lock:
            D, I = faiss_index.search(q_emb.reshape(1, -1), req.vector_size)
        faiss_scores = {faiss_doc_ids[int(idx)]: float(D[0][i]) for i, idx in enumerate(I[0]) if idx >= 0}
        max_es = max(es_scores.values()) if es_scores else 1.0
        min_f, max_f = (min(faiss_scores.values()), max(faiss_scores.values())) if faiss_scores else (0, 1)
        combined = []
        for did in set(es_scores) & set(faiss_scores):
            se = es_scores[did] / max_es
            sf = 0 if max_f == min_f else (faiss_scores[did] - min_f) / (max_f - min_f)
            score = (req.weight_text * se + req.weight_vector * sf) / (req.weight_text + req.weight_vector)
            combined.append((did, score))
        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [d for d, _ in combined[:req.final_k]] or [d for d, _ in sorted(es_scores.items(), key=lambda x: x[1], reverse=True)[:req.final_k]]
        docs = es.mget(index=INDEX_NAME, body={"ids": top_ids})["docs"]
        results = [SearchResult(id=d["_id"], record=DocumentMedical.parse_obj(d["_source"]), score=dict(combined).get(d["_id"], 0.0)) for d in docs if d.get("found")]
        if req.detailed and results:
            logits = reranker.predict([(req.query, r.record.texte_brut or "") for r in results])
            for r, logit in zip(results, logits):
                r.score = sigmoid(logit)
            results.sort(key=lambda x: x.score, reverse=True)
        return SearchResponse(results=results, note="cross-encoder applied" if req.detailed else "bm25+faiss only")
    except Exception:
        logger.error("Search error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="internal error")
