# recherche
# Medical Search API

**Hybrid semantic search** over medical documents combining:

* **Elasticsearch BM25** for strong keyword recall
* **FAISS vector search** for semantic similarity
* **Cross‑Encoder reranking** (optional) for extra precision

---

## Endpoints

| Route                                 | Purpose                                                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`POST /add_document`** *(dev‑only)* | Bulk‑insert test docs (title, content, summary). **Not used in prod**—real docs are indexed by your extraction pipeline. |
| **`POST /hybrid_search`**             | Query the medical index. Combines BM25 + FAISS and, if `detailed=true`, reranks with a Cross‑Encoder.                    |

---

## Quick Start (Local)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Open docs
open http://localhost:8000/docs
```

**Docker**

```bash
docker build -t medical-search-api .
docker run -d -p 8000:8000 medical-search-api
```

---

## Front‑End Search Request

Fill `NEXT_PUBLIC_API_URL` in your `.env.local` (e.g. `http://localhost:8000`).

```ts
// example inside a React/Next.js component
const body = {
  query: userInput,        // « fever », « chest pain », etc.
  text_size: 50,           // # BM25 candidates
  vector_size: 50,         // # FAISS candidates
  final_k: 5,              // results to return
  weight_text: 1.0,        // BM25 weight
  weight_vector: 1.0,      // FAISS weight
  detailed: preciseMode,   // boolean toggle for Cross‑Encoder
};

const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/hybrid_search`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const data = await res.json();
setResults(data.results); // [{ id, title, summary, score, … }]
```

---

Ready!  Spin up the API, point your front‑end to `/hybrid_search`, and you’ll get ranked medical records with a relevance `score` for display.
