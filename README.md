# Tender Intelligence Console

A production-style **tender QA + bid/no-bid copilot** for European procurement PDFs.

Built for an anonymized industrial client (codename **Aurora Works**) who wanted faster, evidence‑based tender reviews without losing traceability.

## Demo

[Watch the demo](media/DEMO%20AIN%20SCARED.mp4)

## What it does

- **Ingest tender PDFs** (drag & drop into chat or batch import)
- **Parse tenders into sections** (buyer, scope, lots, deadlines, requirements, submissions) with collapsible UI
- **Assess readiness** (can we qualify/deliver) using tender requirements plus your **company profile, compliance, and delivery history** (Postgres)
- **Find similar tenders** with evidence-backed matches and snippets
- **Answer with traceable citations** (`file#page`) for every factual claim

## Why it matters

Bid teams waste hours manually scanning PDFs and still miss constraints.
This console turns a tender into a **decision‑ready brief** with explicit evidence and next steps.

## Highlights (portfolio‑worthy)

- **LangGraph tool-calling agent**: model routes to the right tool (breakdown, readiness, similarity, chat history)
- **Evidence-first answers**: every factual claim is grounded in retrieved excerpts with citations
- **Semantic chunking** + query rewrite for higher retrieval precision on long documents
- **Server-side chat memory** (threaded sessions) + retrieval controls (depth, source filter)
- **Full product surface**: API + web UI + ingestion pipeline + DB seed + Docker Compose

## Tech stack

- **Backend**: FastAPI, LangChain, LangGraph
- **RAG**: Pinecone (vector DB), OpenAI embeddings
- **LLM**: OpenAI Chat Completions (configurable)
- **DB**: Postgres (company profile + delivery history)
- **Frontend**: Next.js
- **Ops**: Docker Compose, `.env` configuration, structured logging

## Quickstart (Docker)

1) Create `agent/src/.env` with your secrets:

```bash
OPENAI_API_TOKEN=...
PINECONE_TOKEN=...
PINECONE_INDEX=...
PINECONE_NAMESPACE=tenders
```

2) Start the stack:

```bash
cd agent
docker compose up --build
```

3) Open:

- Web UI: `http://localhost:3000`
- API: `http://localhost:8000`

## Typical workflow

1) Put tender PDFs into `agent/src/data/tenders/raw/` (optional if you attach via chat)
2) Chunk + embed + upsert to Pinecone (CLI)
3) Ask questions like:

- “Break down the tender by scope, deadlines, requirements, and submission details.”
- “What are the qualification thresholds and what evidence do we need?”
- “Find similar tenders and explain the match.”
- “Can we qualify? Give a decision-ready assessment with gaps and next steps.”

## Architecture (high level)

`Next.js UI` → `FastAPI` → `LangGraph Agent (tools)` → `{Pinecone, Postgres, PDF parser}` → response (with sources)

## Repo layout

- `agent/` — the full application (API, UI, ingestion scripts, DB seed)

For detailed CLI commands (chunking, embeddings, Pinecone upload), see `agent/README.md`.

## Data & privacy notes

- Secrets are intentionally excluded from git (`.env` is ignored).
- Intermediate artifacts (chunk/embedding dumps) should not be committed.
- The company profile and deliveries included here are **synthetic demo data** to showcase readiness checks.

## License

Apache‑2.0 (see `LICENSE`).
