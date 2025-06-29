import os
import json
import uuid
import asyncio
import logging
import traceback
import time
from math import exp
from typing import List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configuration
INDEX_NAME = "dossiers_medicaux"
EMBEDDING_DIM = 384
FAISS_INDEX_PATH = "faiss_hnsw.index"
FAISS_IDS_PATH = "faiss_ids.json"
RERANK_CANDIDATES = 5

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_search_app")

# FastAPI initialization with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Elasticsearch client
es = Elasticsearch(
    "http://34.224.105.165:9200",
    request_timeout=30,
    max_retries=10,
    retry_on_timeout=True,
)

# SentenceTransformer models
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cuda"
    )
except Exception:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# FAISS index and document ID store
faiss_index: faiss.IndexFlat = None
faiss_doc_ids: List[str] = []
faiss_lock = asyncio.Lock()

# ----------------------
# Helper functions
# ----------------------

def sigmoid(x: float) -> float:
    """Convert logit to probability between 0 and 1."""
    return 1.0 / (1.0 + exp(-x))

# ----------------------
# Elasticsearch & FAISS setup
# ----------------------

def create_index() -> None:
    """Create Elasticsearch index with dense_vector and synonym analyzer."""
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
                "title": {"type": "text", "analyzer": "synonym_analyzer"},
                "content": {"type": "text", "analyzer": "synonym_analyzer"},
                "summary": {"type": "text", "analyzer": "synonym_analyzer"},
                "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIM}
            }
        }
    }
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        logger.info(f"Created index: {INDEX_NAME}")


def load_faiss() -> None:
    """Load or initialize the FAISS index and document ID list."""
    global faiss_index, faiss_doc_ids
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_IDS_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_IDS_PATH, "r") as f:
            faiss_doc_ids = json.load(f)
        logger.info("FAISS index loaded from disk.")
    else:
        faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_doc_ids = []
        logger.info("Initialized new FAISS index.")


def save_faiss() -> None:
    """Persist the FAISS index and doc IDs to disk."""
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "w") as f:
        json.dump(faiss_doc_ids, f)
    logger.info("FAISS index saved to disk.")

# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    create_index()
    load_faiss()

# ----------------------
# Pydantic models
# ----------------------

class Document(BaseModel):
    title: str
    content: str
    summary: str


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
    title: str
    content: str
    summary: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    note: Optional[str] = None

# ----------------------
# API endpoints
# ----------------------

@app.post("/add_document")
async def add_documents(docs: List[Document]) -> dict:
    """
    Add documents to Elasticsearch and FAISS index.
    """
    actions = []
    async with faiss_lock:
        for doc in docs:
            text = f"{doc.content} {doc.summary}"
            emb = np.array(model.encode(text), dtype=np.float32)
            emb /= np.linalg.norm(emb)
            doc_id = str(uuid.uuid4())
            actions.append({
                "_index": INDEX_NAME,
                "_id": doc_id,
                "_source": {
                    "title": doc.title,
                    "content": doc.content,
                    "summary": doc.summary,
                    "embedding": emb.tolist(),
                }
            })
            faiss_index.add(emb.reshape(1, -1))
            faiss_doc_ids.append(doc_id)
        success, errors = bulk(es, actions)
        save_faiss()
    return {"inserted": len(docs), "errors": errors}


@app.post("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest) -> SearchResponse:
    """
    Perform hybrid search: BM25 + FAISS, then optional Cross-Encoder rerank.
    """
    try:
        # Encode query
        q_emb = np.array(model.encode(req.query), dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb)

        # Elasticsearch BM25 + kNN query
        es_query = {
            "size": max(req.text_size, req.vector_size),
            "query": {"bool": {"should": [
                {"multi_match": {
                    "query": req.query,
                    "fields": ["title","content","summary"],
                    "analyzer": "synonym_analyzer",
                    "fuzziness": "AUTO",
                    "boost": req.weight_text
                }},
                {"knn": {"embedding": {"vector": q_emb.tolist(), "k": req.vector_size, "boost": req.weight_vector}}}
            ]}}
        }
        try:
            res = es.search(index=INDEX_NAME, body=es_query)
        except Exception:
            # Fallback to text-only
            text_q = {"size": req.text_size, "query": {"multi_match": {"query": req.query, "fields": ["title","content","summary"], "fuzziness": "AUTO"}}}
            res = es.search(index=INDEX_NAME, body=text_q)
        es_scores = {h["_id"]: h["_score"] for h in res["hits"]["hits"]}

        # FAISS search
        async with faiss_lock:
            D, I = faiss_index.search(q_emb.reshape(1, -1), req.vector_size)
        faiss_scores = {faiss_doc_ids[int(idx)]: float(D[0][i]) for i, idx in enumerate(I[0]) if idx >= 0}

        # Combine scores
        max_es = max(es_scores.values()) if es_scores else 1.0
        vals_f = list(faiss_scores.values()) or [0.0,1.0]
        min_f, max_f = min(vals_f), max(vals_f)
        total_weight = req.weight_text + req.weight_vector
        combined = []
        for did in set(es_scores) & set(faiss_scores):
            norm_es = es_scores[did] / max_es
            norm_f = (faiss_scores[did] - min_f) / (max_f - min_f) if max_f > min_f else faiss_scores[did]
            score = (req.weight_text * norm_es + req.weight_vector * norm_f) / total_weight
            combined.append((did, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Top-k selection
        combined_dict = dict(combined)
        top_ids = list(combined_dict.keys())[:req.final_k] if combined_dict else []
        if not top_ids:
            # Fallback text-only sort
            sorted_es = sorted(es_scores.items(), key=lambda x: x[1], reverse=True)
            top_ids = [did for did, _ in sorted_es[:req.final_k]]
            combined_dict = {did: min(1.0, es_scores[did]/max_es) for did in top_ids}

        # Retrieve documents
        docs_resp = es.mget(index=INDEX_NAME, body={"ids": top_ids})
        results: List[SearchResult] = []
        for d in docs_resp.get("docs", []):
            if d.get("found"):
                src = d["_source"]
                results.append(SearchResult(
                    id=d["_id"],
                    title=src["title"],
                    content=src["content"],
                    summary=src["summary"],
                    score=combined_dict.get(d["_id"], 0.0)
                ))

        # Quick mode
        if not req.detailed:
            return SearchResponse(results=results, note="bm25+faiss only")

        # Cross-encoder rerank
        raw_scores = reranker.predict([(req.query, doc.content) for doc in results])
        for idx, raw in enumerate(raw_scores):
            results[idx].score = sigmoid(raw)
        results.sort(key=lambda x: x.score, reverse=True)

        return SearchResponse(results=results, note="cross-encoder applied")

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"hybrid_search ERROR:\n{tb}")
        raise HTTPException(status_code=500, detail="Internal server error")
