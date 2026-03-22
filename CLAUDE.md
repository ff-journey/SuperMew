# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
uv sync                          # Install dependencies
docker compose up -d             # Start Milvus (etcd + minio + standalone + attu)
```

### Run the server
```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
# or
uv run python backend/app.py
```

### Run LangSmith evaluation
```bash
uv run python test_langsmith_eval.py
```

The server serves the frontend at `http://127.0.0.1:8000/` and API docs at `http://127.0.0.1:8000/docs`.

## Environment Variables

Create a `.env` in the project root:

```env
# LLM
ARK_API_KEY=...
MODEL=...
BASE_URL=https://your-llm-endpoint/v1
EMBEDDER=...
GRADE_MODEL=gpt-4.1          # optional, defaults to gpt-4.1

# Rerank (optional, degrades gracefully)
RERANK_MODEL=...
RERANK_BINDING_HOST=...
RERANK_API_KEY=...

# Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=embeddings_collection

# Auto-merging (optional)
AUTO_MERGE_ENABLED=true
AUTO_MERGE_THRESHOLD=2
LEAF_RETRIEVE_LEVEL=3

# Tools (optional)
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=...
```

## Architecture

**Stack**: FastAPI + LangChain/LangGraph Agent + Milvus + Vue 3 (CDN SPA)

### Backend module structure (`backend/`)

All backend modules import each other without the `backend.` prefix — they run with `backend/` as the working directory.

| File | Role |
|---|---|
| `app.py` | FastAPI app factory; mounts `api.py` router and serves `frontend/` as static files |
| `api.py` | REST endpoints: `/chat`, `/chat/stream` (SSE), `/sessions/*`, `/documents/*` |
| `agent.py` | `ConversationStorage` (JSON-backed), `chat_with_agent`, `chat_with_agent_stream`; summary compression at 50+ messages |
| `tools.py` | `search_knowledge_base` (@tool), `get_current_weather`; cross-thread RAG step emission globals |
| `rag_pipeline.py` | LangGraph `StateGraph` (RAGState): `retrieve_initial` → `grade_documents` → conditional → `rewrite_question` → `retrieve_expanded` → END |
| `rag_utils.py` | `retrieve_documents` (hybrid+rerank+auto-merge), `step_back_expand`, `generate_hypothetical_document` |
| `milvus_client.py` | `MilvusManager`: collection init, `hybrid_retrieve` (Dense+Sparse+RRF), `dense_retrieve` (fallback) |
| `embedding.py` | `EmbeddingService`: dense embedding API calls + jieba-based BM25 sparse vector generation |
| `document_loader.py` | PDF/Word/Excel loading; 3-level sliding window chunking (L1/L2/L3) with hierarchy metadata |
| `milvus_writer.py` | Writes L3 leaf chunks (dense+sparse vectors) to Milvus |
| `parent_chunk_store.py` | `ParentChunkStore`: JSON-backed DocStore for L1/L2 parent chunks used in auto-merging |
| `schemas.py` | Pydantic request/response models |

### Data files (`data/`)
- `customer_service_history.json` — conversation history keyed by `user_id/session_id`
- `parent_chunks.json` — L1/L2 parent chunk store for auto-merging
- `documents/` — uploaded source files

### Frontend (`frontend/`)
Vue 3 CDN SPA (`index.html` + `script.js` + `style.css`). Parses SSE stream manually via `ReadableStream`. Implements a thinking-state-machine: Idle → Thinking → Thinking+RAG steps → Streaming text, all within a single message bubble.

## Key Architectural Patterns

### Cross-thread RAG step emission
LangChain runs sync tools in a `ThreadPoolExecutor`, making `asyncio.Queue` inaccessible from the worker thread. The solution:
1. `set_rag_step_queue()` captures `asyncio.get_running_loop()` into `_RAG_STEP_LOOP` on the main thread.
2. `emit_rag_step()` calls `_RAG_STEP_LOOP.call_soon_threadsafe(queue.put_nowait, step)` from the worker thread.
3. In `chat_with_agent_stream`, a `_RagStepProxy` wraps the queue so steps are tagged `{"type": "rag_step"}` before entering the unified output queue.

### Streaming architecture (unified output queue)
`chat_with_agent_stream` creates one `asyncio.Queue`. A background `_agent_worker` task pushes `{"type": "content"}` chunks. RAG steps arrive via `call_soon_threadsafe`. The main generator loop pulls from the queue and yields SSE. On `GeneratorExit` (client abort), `agent_task.cancel()` is called explicitly for deterministic cleanup.

### RAG pipeline (LangGraph)
`rag_pipeline.py` builds a `StateGraph` with `RAGState`. Flow:
- `retrieve_initial` → hybrid search (L3 leaf chunks) + auto-merge
- `grade_documents` → structured output binary relevance score (`yes` → END, `no` → rewrite)
- `rewrite_question` → LLM routes to `step_back` / `hyde` / `complex` strategy
- `retrieve_expanded` → re-retrieves with expanded query, deduplicates, re-indexes RRF ranks

### Hybrid retrieval + auto-merging (`rag_utils.py`)
`retrieve_documents` filters by `chunk_level == LEAF_RETRIEVE_LEVEL`, runs `hybrid_retrieve` (or `dense_retrieve` on fallback), optionally calls Jina Rerank API, then runs `_merge_to_parent_level` iteratively (L3→L2→L1): if `threshold` or more leaf siblings share a parent, replace them with the parent chunk from `ParentChunkStore`.

### Leaf-only vector storage
Only L3 chunks are written to Milvus. L1/L2 parent chunks are stored in `parent_chunks.json` via `ParentChunkStore`. This avoids redundant vectors while preserving context aggregation via auto-merging at query time.

### SSE event types
| `type` | Content |
|---|---|
| `content` | Text token (typewriter effect) |
| `rag_step` | Real-time retrieval step `{icon, label, detail}` |
| `trace` | Full `rag_trace` dict (sent after answer completes) |
| `error` | Error message string |
| `[DONE]` | Stream end marker |
