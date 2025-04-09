import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import uuid
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Elasticsearch index name
INDEX_NAME = "dossiers_medicaux"

app = FastAPI()

# Connect to Elasticsearch (ensure you're using the official distribution)
es = Elasticsearch(
    "http://127.0.0.1:9200",
    request_timeout=30,
    max_retries=10,
    retry_on_timeout=True
)

# Initialize SentenceTransformer model (384-dimensional embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# Define Elasticsearch mapping using dense_vector for embeddings.
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "summary": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIM}
        }
    }
}

@app.on_event("startup")
async def startup_event():
    # Create Elasticsearch index if it does not exist.
    print("Checking if Elasticsearch index exists...")
    if not es.indices.exists(index=INDEX_NAME):
        try:
            print(f"Index {INDEX_NAME} not found. Creating...")
            es.indices.create(index=INDEX_NAME, body=mapping)
            print(f"Index {INDEX_NAME} created successfully.")
        except Exception as e:
            print("Error creating index:", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    else:
        print(f"Index {INDEX_NAME} already exists.")
    # Initialize global FAISS index and document ID mapping.
    initialize_faiss_index()

# --- Global FAISS index setup ---
faiss_index = None
faiss_doc_ids = []  # Maps FAISS index positions to Elasticsearch document IDs

def initialize_faiss_index():
    global faiss_index, faiss_doc_ids
    # Using a flat index with Inner Product (vectors are normalized for cosine similarity)
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss_doc_ids = []
    print("FAISS index initialized.")

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# --- Data model ---
class Document(BaseModel):
    title: str
    content: str
    summary: str

@app.post("/add_document")
async def add_document(docs: List[Document]):
    """
    For each document, generate an embedding using (content + summary),
    then insert the document into Elasticsearch and update the FAISS index.
    """
    global faiss_index, faiss_doc_ids
    print("=== Adding documents (Hybrid) ===")
    es_actions = []
    for doc in docs:
        text = f"{doc.content} {doc.summary}"
        embedding = model.encode(text)
        embedding = np.array(embedding, dtype=np.float32)
        embedding = normalize_vector(embedding)
        doc_id = str(uuid.uuid4())
        doc_source = {
            "title": doc.title,
            "content": doc.content,
            "summary": doc.summary,
            "embedding": embedding.tolist()
        }
        action = {"_index": INDEX_NAME, "_id": doc_id, "_source": doc_source}
        es_actions.append(action)
        # Update FAISS index and document ID mapping.
        faiss_index.add(np.expand_dims(embedding, axis=0))
        faiss_doc_ids.append(doc_id)
        print(f"Document '{doc.title}' added with ID {doc_id}")
    try:
        success, errors = bulk(es, es_actions)
        print(f"Bulk insert: {success} documents inserted. Errors: {errors}")
    except Exception as e:
        print("Bulk insert error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    return {"inserted": len(docs), "success": success}

# --- Hybrid Search Endpoint ---
class HybridSearchRequest(BaseModel):
    query: str
    text_size: int = 50    # Number of candidate docs to retrieve by text search
    vector_size: int = 50  # Number of candidate docs to retrieve by FAISS
    final_k: int = 5       # Final number of results to return
    weight_text: float = 1.0   # Weight for normalized text score
    weight_vector: float = 1.0 # Weight for vector score

@app.post("/hybrid_search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Performs hybrid search by:
      1. Running a text search on Elasticsearch (BM25).
      2. Running FAISS ANN search for cosine similarity.
      3. Normalizing and combining scores additively.
    Returns the top 'final_k' documents with id, title, content, summary, and combined score.
    """
    global faiss_index, faiss_doc_ids
    print("=== Hybrid Search ===")
    print(f"Query received: {request.query}")

    # Step 1: FAISS vector search
    query_embedding = model.encode(request.query)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    query_embedding = normalize_vector(query_embedding)
    vector_k = request.vector_size
    D, I = faiss_index.search(np.expand_dims(query_embedding, axis=0), vector_k)
    faiss_candidates = {}
    for i, idx in enumerate(I[0]):
        idx = int(idx)  # Ensure the index is an integer.
        if idx < 0 or idx >= len(faiss_doc_ids):
            continue
        doc_id = faiss_doc_ids[idx]
        faiss_candidates[doc_id] = float(D[0][i])
    print(f"FAISS found {len(faiss_candidates)} candidate IDs.")

    # Step 2: Text search on Elasticsearch
    if faiss_candidates:
        text_query = {
            "size": request.text_size,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": request.query,
                            "fields": ["title", "content", "summary"],
                            "fuzziness": "AUTO"
                        }
                    },
                    "filter": {"terms": {"_id": list(faiss_candidates.keys())}}
                }
            }
        }
    else:
        text_query = {
            "size": request.text_size,
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["title", "content", "summary"],
                    "fuzziness": "AUTO"
                }
            }
        }
    try:
        text_response = es.search(index=INDEX_NAME, body=text_query)
    except Exception as e:
        print("Error during Elasticsearch text search:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    text_scores = {}
    for hit in text_response["hits"]["hits"]:
        doc_id = hit["_id"]
        text_scores[doc_id] = hit["_score"]
    print(f"Elasticsearch text search returned {len(text_scores)} candidate documents.")

    # Step 3: Normalize text scores and combine scores additively.
    combined_candidates = []
    if text_scores:
        max_text_score = max(text_scores.values())
    else:
        max_text_score = 1.0

    if not faiss_candidates:
        print("No FAISS candidates found, using text search results only.")
        # Use normalized text score alone.
        for doc_id, score in text_scores.items():
            normalized_text = score / max_text_score
            combined_candidates.append((doc_id, request.weight_text * normalized_text))
    else:
        for doc_id in set(faiss_candidates.keys()).intersection(text_scores.keys()):
            normalized_text = text_scores[doc_id] / max_text_score
            # Combine scores additively.
            combined_score = request.weight_text * normalized_text + request.weight_vector * faiss_candidates[doc_id]
            combined_candidates.append((doc_id, combined_score))
    combined_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = combined_candidates[: request.final_k]
    print("Top candidates after combining scores:")
    for rank, (doc_id, score) in enumerate(top_candidates, start=1):
        print(f"{rank}. Doc ID: {doc_id} with combined score: {score:.4f}")

    # Step 4: Retrieve full documents from Elasticsearch
    final_docs = []
    if top_candidates:
        doc_ids = [doc_id for doc_id, _ in top_candidates]
        try:
            final_response = es.mget(index=INDEX_NAME, body={"ids": doc_ids})
        except Exception as e:
            print("Error during mget:", str(e))
            raise HTTPException(status_code=500, detail=str(e))
        for doc in final_response.get("docs", []):
            if doc.get("found"):
                source = doc["_source"]
                final_docs.append({
                    "id": doc["_id"],
                    "title": source.get("title"),
                    "content": source.get("content"),
                    "summary": source.get("summary"),
                    "combined_score": next((score for id_, score in top_candidates if id_ == doc["_id"]), None)
                })

    # Print the top candidate details in a JSON-like format.
    print("Final Top Candidates:")
    print(json.dumps(final_docs, indent=2))
    return {"results": final_docs}


