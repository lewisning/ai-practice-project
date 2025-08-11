# Tucows RAG Knowledge Assistant (Python 3.10 + Qdrant + OpenAI)

Single endpoint: `POST /resolve-ticket`
Input a customer support ticket text and output an MCP-compliant JSON:

```json
{
  "answer": "...",
  "references": ["Doc · Section · anchor-id", "..."],
  "action_required": "..."
}
```

## ✨ Features

* Multi-retrieval: Vector search (Qdrant, cosine) + BM25 fallback → fusion reranking
* Citation tracking: Returns snippet references as `Doc · Section · anchor-id`
* Compliance: Strict JSON (Pydantic validation + retry)
* Action routing: Rule-based classification (e.g., suspension/abuse → `escalate_to_abuse_team`)
* Engineering quality: Docker Compose, `/metrics` endpoint

## 🧱 Tech Stack

Python 3.10, FastAPI, Qdrant, OpenAI Embeddings + Chat, rank-bm25, structlog, pytest

## 🚀 Quick Start

### 1) Prepare Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### 2) Launch with Docker Compose

```bash
docker compose up -d --build
```

* API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Metrics: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)
* Qdrant: [http://127.0.0.1:6333/collections](http://127.0.0.1:6333/collections)

### 3) Demo

```bash
# Instantly ingest single sample document data
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"doc":"Policy: Domain Suspension Guidelines","section":"4.2","anchor_id":"para-17","text":"A domain may be suspended due to invalid WHOIS information. To reactivate, update WHOIS and provide proof of registrant identity.","product":"domains","lang":"en"}'
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"doc":"WHOIS Requirements","section":"2.1","anchor_id":"para-05","text":"Registrants must keep WHOIS contact details accurate. Verification emails must be completed within the specified window to avoid suspension.","product":"domains","lang":"en"}'

# Ingest single document with file upload
curl -s -X POST http://127.0.0.1:8000/ingest-file -F "file=@data/your_file" -F "product=domains" -F "lang=en" | jq

# Ingest multiple documents from a directory
curl -s -X POST http://127.0.0.1:8000/ingest-path -H "Content-Type: application/json" -d '{"path":"/app/data","product":"domains","lang":"en"}' | jq

# Merged search with BM25 + Qdrant vector search
curl -s -X POST http://127.0.0.1:8000/search_merged -H "Content-Type: application/json" -d '{"query":"reactivate a suspended domain due to invalid WHOIS","top_k":5,"product":"domains","lang":"en"}' | jq

# End-to-end MCP-compliant answer
curl -s -X POST http://127.0.0.1:8000/resolve-ticket -H "Content-Type: application/json" -d '{"ticket_text":"My domain was suspended and I didn’t get any notice. How can I reactivate it?","top_k":8}' | jq
```

## 🥪 Testing

```bash
pytest -q
# or with coverage
coverage run -m pytest -q && coverage report -m
```

## ⚙️ Configuration & Parameters

* `.env`:

  * `OPENAI_API_KEY`: Your OpenAI key
  * `OPENAI_GPT_NAME`: 'gpt-4o-mini' or your preferred model
  * `EMBEDDING_MODEL`: `text-embedding-3-small` (1536 dimensions)
  * `QDRANT_HOST/PORT/COLLECTION`: defaults to `qdrant:6333 / kb_chunks`
* Key parameters: `VECTOR_TOPK=30`, `BM25_TOPK=20`, `MAX_CTX_SNIPPETS=8`, `alpha=0.7`

## 🧩 Architecture Overview

```
Ticket --> /resolve-ticket --> Orchestrator
   |                                |
   |                         search_merged()
   |                         ---- semantic (Qdrant)
   |                         ---- BM25 (in-memory)
   |                         --> score fusion --> top snippets
   |                                |
   |                       MCP Prompt (system + schema + context)
   |                                |
   |                      OpenAI Chat (json only)
   |                                |
   |                   Pydantic validate + action rules
   v
{answer, references, action_required}
```

## 📙 Adding Knowledge Base Documents

Place Markdown/HTML/PDF files in `data/` and run your ingest pipeline. For example:

* Use `/ingest` API for quick additions
* Use `/ingest-file` for single file uploads
* Use `/ingest-path` to ingest all files in `/data` directory

## 🩹 Troubleshooting

* `OpenAIError: api_key ...`: Ensure `.env` is loaded into the container via `env_file`
* `proxies` error: Remove `OPENAI_*_PROXY` and use standard `HTTPS_PROXY/HTTP_PROXY` instead
* `/qdrant/health` returns false: Check port 6333, network connectivity, and Docker Compose dependencies
* Embedding dimension mismatch: `EMBEDDING_DIM` must match model dimensions (small=1536, large=3072)

## Author

Developed and maintained by [Xuyang (Lewis) Ning](https://github.com/lewisning)

---