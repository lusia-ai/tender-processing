# PDF intake starter

Minimal tooling to fetch a PDF (local path or URL), parse first pages, and print previews to stdout. Intended as the front-end of a future LangChain/LangGraph RAG pipeline.

## Quickstart (venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"

# Parse a URL (saved to data/tenders/raw by default)
python -m pdf_agent.cli --url "https://example.com/sample.pdf" --max-pages 2
# Or use a temp file instead of saving:
python -m pdf_agent.cli --url "https://example.com/sample.pdf" --temp

# Parse a local file
python -m pdf_agent.cli --pdf /path/to/tender.pdf --max-pages 3 --preview-chars 600
```

- `--max-pages <= 0` parses all pages.
- Prints per-page character counts and the first N characters of text (whitespace-collapsed).
- URL downloads default to `data/tenders/raw/`; override with `--save-dir ...`.

## Docker
```bash
docker build -t pdf-agent .
docker run --rm pdf-agent --url "https://example.com/sample.pdf" --max-pages 2
```

## Batch chunking for RAG
- Prepare chunks from all PDFs (defaults to `data/tenders/raw/`):
```bash
PYTHONPATH=. .venv/bin/python -m pdf_agent.batch \
  --input-dir data/tenders/raw \
  --output data/tenders/processed/chunks.jsonl \
  --chunk-size 600 \
  --overlap 100 \
  --preserve-whitespace
```
- JSONL includes `type`, `source`, `chunk_id`, `page`, `pages`, `order`, `text`, `chars`.
- Semantic chunking via LLM (detecting topic shifts between blocks):
```bash
PYTHONPATH=. .venv/bin/python -m pdf_agent.batch \
  --input-dir data/tenders/raw \
  --output data/tenders/processed/chunks.jsonl \
  --strategy semantic-llm \
  --min-tokens 600 \
  --max-tokens 1200 \
  --log-llm
```
Semantic-llm requires `OPENAI_API_TOKEN` (or `OPENAI_API_KEY`) and `OPENAI_MODEL` in the environment or `.env`.

## Lot extraction (description <-> amounts)
- Creates lot/task-level records with descriptions and amounts when found:
```bash
PYTHONPATH=. .venv/bin/python -m pdf_agent.parsing.lot_extractor \
  --input-dir data/tenders/raw \
  --output data/tenders/processed/lots.jsonl
```
- Uses Camelot (stream) for tables + regex for sections II.2/III.1.3. Records are saved as JSONL with `type=lot`.

## Embeddings and Pinecone
1) Install extra dependencies (OpenAI + local models + Pinecone client + dotenv):
```bash
pip install -r requirements-embeddings.txt
```
2) Put keys in environment variables (you can use `.env` - it is auto-loaded):
   - `OPENAI_API_TOKEN` (or `OPENAI_API_KEY`)
   - `OPENAI_EMBED_MODEL`
   - `OPENAI_EMBED_DIM`
   - `PINECONE_TOKEN` (or `PINECONE_API_KEY`) if sending to Pinecone
3) Compute embeddings and (optionally) send to Pinecone:
- OpenAI (model from `OPENAI_EMBED_MODEL`):
```bash
PYTHONPATH=. .venv/bin/python -m pdf_agent.embed \
  --chunks-file data/tenders/processed/chunks.jsonl \
  --output data/tenders/processed/embeddings_openai.jsonl \
  --provider openai \
  --batch-size 32
# same for lots (description <-> amount):
PYTHONPATH=. .venv/bin/python -m pdf_agent.embed \
  --chunks-file data/tenders/processed/lots.jsonl \
  --output data/tenders/processed/embeddings_lots.jsonl \
  --provider openai \
  --batch-size 32
# with Pinecone upload:
PYTHONPATH=. .venv/bin/python -m pdf_agent.embed \
  --chunks-file data/tenders/processed/chunks.jsonl \
  --output data/tenders/processed/embeddings_openai.jsonl \
  --provider openai \
  --pinecone-index your-index \
  --namespace tenders
PYTHONPATH=. .venv/bin/python -m pdf_agent.embed \
  --chunks-file data/tenders/processed/lots.jsonl \
  --output data/tenders/processed/embeddings_lots.jsonl \
  --provider openai \
  --pinecone-index your-index \
  --namespace tenders
```
- Local (SentenceTransformers, e.g. `intfloat/multilingual-e5-base` or `BAAI/bge-small-en-v1.5`):
```bash
PYTHONPATH=. .venv/bin/python -m pdf_agent.embed \
  --chunks-file data/tenders/processed/chunks.jsonl \
  --output data/tenders/processed/embeddings_bge.jsonl \
  --provider bge \
  --model intfloat/multilingual-e5-base \
  --batch-size 16
```
Embeddings file is JSONL like `{id, embedding, metadata}`; if `--pinecone-index` is set, records are sent to Pinecone immediately (key from `PINECONE_API_KEY` or the flag).

## Console RAG agent (LangGraph + Pinecone)
```bash
export PYTHONPATH="$PWD"
source .venv/bin/activate

python -m pdf_agent.agent \
  --query "What is being procured and who is the buyer?" \
  --pinecone-index port \
  --namespace tenders \
  --top-k 6 \
  --source-filter ted_812-2018_EN.pdf   # optionally limit search to one file
```
- Uses keys from `.env` (`OPENAI_API_TOKEN`, `PINECONE_TOKEN`); override with `--api-key` and `--pinecone-api-key`.
- Model settings are required via env (`OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, `OPENAI_EMBED_DIM`); CLI flags override if provided.
- Graph steps: embed query -> Pinecone retrieval (top_k) -> answer generation with sources `file#page`.
- By default the agent searches `type=lot` first (description <-> amount). Disable with `--no-prefer-lots`.

## API + UI (Docker Compose)
- Set variables in `.env`:
  - `OPENAI_API_TOKEN` (or `OPENAI_API_KEY`)
  - `PINECONE_TOKEN` (or `PINECONE_API_KEY`)
  - `PINECONE_INDEX=port` (your index)
  - `PINECONE_NAMESPACE=tenders`
  - `OPENAI_MODEL`
  - `OPENAI_EMBED_MODEL`
  - `OPENAI_EMBED_DIM`
  - optional: `PREFER_LOTS=true`
  - optional: `ENABLE_CHECKPOINTING=true`
  - optional: `CHECKPOINT_TTL_SECONDS=0`
  - optional: `CHECKPOINT_MAX_THREADS=0`
- Run:
```bash
docker compose up --build
```
- API: `http://localhost:8000/ask` (POST `{"query": "...", "top_k":6, "source_filter":"..."}`)
- Chat: `http://localhost:8000/chat` (POST `{"message": "...", "thread_id":"...", "top_k":6, "source_filter":"..."}`)
- Web UI (Next.js): `http://localhost:3000` - question form, top_k, and file filter.

## Next steps
- Swap previews for chunked text ready for embeddings.
- Add structured field extraction (currency, dates, incoterms, standards).
- Wire Pinecone + LangChain/LangGraph nodes for ingest and querying.
