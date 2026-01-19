# PDF intake starter

Minimal tooling to fetch a PDF (local path or URL), parse first pages, and print previews to stdout. Intended as the front-end of a future LangChain/LangGraph RAG pipeline.

## Quickstart (venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"

# Parse a URL (saved to src/data/tenders/raw by default)
python -m pdf_agent.cli --url "https://example.com/sample.pdf" --max-pages 2
# Or use a temp file instead of saving:
python -m pdf_agent.cli --url "https://example.com/sample.pdf" --temp

# Parse a local file
python -m pdf_agent.cli --pdf /path/to/tender.pdf --max-pages 3 --preview-chars 600
```

- `--max-pages <= 0` parses all pages.
- Prints per-page character counts and the first N characters of text (whitespace-collapsed).
- URL downloads default to `src/data/tenders/raw/`; override with `--save-dir ...`.

## Docker
```bash
docker build -t pdf-agent .
docker run --rm pdf-agent --url "https://example.com/sample.pdf" --max-pages 2
```

## Batch chunking for RAG
- Подготовить чанки из всех PDF (по умолчанию из `src/data/tenders/raw/`):
```bash
PYTHONPATH=src .venv/bin/python -m pdf_agent.batch \
  --input-dir src/data/tenders/raw \
  --output src/data/tenders/processed/chunks.jsonl \
  --chunk-size 600 \
  --overlap 100 \
  --preserve-whitespace
```
- JSONL содержит поля `type`, `source`, `chunk_id`, `page`, `pages`, `order`, `text`, `chars`.
- Семантический чанкинг через LLM (определение смены темы между блоками):
```bash
PYTHONPATH=src .venv/bin/python -m pdf_agent.batch \
  --input-dir src/data/tenders/raw \
  --output src/data/tenders/processed/chunks.jsonl \
  --strategy semantic-llm \
  --min-tokens 600 \
  --max-tokens 1200 \
  --llm-model gpt-4o-mini \
  --log-llm
```
Для semantic-llm нужен `OPENAI_API_TOKEN` (или `OPENAI_API_KEY`) в окружении или `src/.env`.

## Извлечение лотов (описание ↔ суммы)
- Создает записи на уровне лотов/заданий с описанием и суммами, если они найдены:
```bash
PYTHONPATH=src .venv/bin/python -m pdf_agent.lot_extractor \
  --input-dir src/data/tenders/raw \
  --output src/data/tenders/processed/lots.jsonl
```
- Внутри используется Camelot (stream) для таблиц + regex для секций II.2/III.1.3. Записи сохраняются как JSONL с `type=lot`.

## Эмбеддинги и Pinecone
1) Установить доп. зависимости (OpenAI + локальные модели + Pinecone клиент + dotenv):
```bash
pip install -r requirements-embeddings.txt
```
2) Положить ключи в переменные окружения (можно в `src/.env` — файл грузится автоматически):
   - `OPENAI_API_TOKEN` (или `OPENAI_API_KEY`)
   - `PINECONE_TOKEN` (или `PINECONE_API_KEY`) если шлём в Pinecone
3) Посчитать эмбеддинги и (опционально) отправить в Pinecone:
- OpenAI (по умолчанию `text-embedding-3-large`):
```bash
PYTHONPATH=src .venv/bin/python -m pdf_agent.embed \
  --chunks-file src/data/tenders/processed/chunks.jsonl \
  --output src/data/tenders/processed/embeddings_openai.jsonl \
  --provider openai \
  --model text-embedding-3-large \
  --dimensions 1024 \
  --batch-size 32
# то же для лотов (связка описание↔сумма):
PYTHONPATH=src .venv/bin/python -m pdf_agent.embed \
  --chunks-file src/data/tenders/processed/lots.jsonl \
  --output src/data/tenders/processed/embeddings_lots.jsonl \
  --provider openai \
  --model text-embedding-3-large \
  --dimensions 1024 \
  --batch-size 32
# с отправкой в Pinecone:
PYTHONPATH=src .venv/bin/python -m pdf_agent.embed \
  --chunks-file src/data/tenders/processed/chunks.jsonl \
  --output src/data/tenders/processed/embeddings_openai.jsonl \
  --provider openai \
  --model text-embedding-3-large \
  --dimensions 1024 \
  --pinecone-index your-index \
  --namespace tenders
PYTHONPATH=src .venv/bin/python -m pdf_agent.embed \
  --chunks-file src/data/tenders/processed/lots.jsonl \
  --output src/data/tenders/processed/embeddings_lots.jsonl \
  --provider openai \
  --model text-embedding-3-large \
  --dimensions 1024 \
  --pinecone-index your-index \
  --namespace tenders
```
- Локально (SentenceTransformers, например `intfloat/multilingual-e5-base` или `BAAI/bge-small-en-v1.5`):
```bash
PYTHONPATH=src .venv/bin/python -m pdf_agent.embed \
  --chunks-file src/data/tenders/processed/chunks.jsonl \
  --output src/data/tenders/processed/embeddings_bge.jsonl \
  --provider bge \
  --model intfloat/multilingual-e5-base \
  --batch-size 16
```
Файл с эмбеддингами — JSONL вида `{id, embedding, metadata}`; если указан `--pinecone-index`, записи сразу отправятся в Pinecone (ключ берётся из `PINECONE_API_KEY` или флага).

## Консольный RAG-агент (LangGraph + Pinecone)
```bash
export PYTHONPATH="$PWD/src"
source .venv/bin/activate

python -m pdf_agent.agent \
  --query "Что за предмет закупки и кто заказчик?" \
  --pinecone-index port \
  --namespace tenders \
  --model gpt-4o-mini \
  --embed-model text-embedding-3-large \
  --dimensions 1024 \
  --top-k 6 \
  --source-filter ted_812-2018_EN.pdf   # опционально ограничить поиск одним файлом
```
- Берёт ключи из `src/.env` (`OPENAI_API_TOKEN`, `PINECONE_TOKEN`), можно переопределить флагами `--api-key` и `--pinecone-api-key`.
- Шаги графа: embed запроса → Pinecone retrieval (top_k) → генерация ответа с указанием источников `file#page`.
- По умолчанию агент сначала ищет в `type=lot` (связка описание↔сумма), отключить: `--no-prefer-lots`.

## API + UI (Docker Compose)
- Настройте переменные в `src/.env`:
  - `OPENAI_API_TOKEN` (или `OPENAI_API_KEY`)
  - `PINECONE_TOKEN` (или `PINECONE_API_KEY`)
  - `PINECONE_INDEX=port` (ваш индекс)
  - `PINECONE_NAMESPACE=tenders`
  - опционально: `PREFER_LOTS=true`
  - опционально: `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, `OPENAI_EMBED_DIM`
- Запуск:
```bash
docker compose up --build
```
- API: `http://localhost:8000/ask` (POST `{"query": "...", "top_k":6, "source_filter":"..."}`)
- Chat: `http://localhost:8000/chat` (POST `{"message": "...", "thread_id":"...", "top_k":6, "source_filter":"..."}`)
- Web UI (Next.js): `http://localhost:3000` — форма для вопроса, top_k и фильтр по файлу.

## Next steps
- Swap previews for chunked text ready for embeddings.
- Add structured field extraction (currency, dates, incoterms, standards).
- Wire Pinecone + LangChain/LangGraph nodes for ingest and querying.
