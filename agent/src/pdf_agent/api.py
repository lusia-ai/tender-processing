from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_agent.agent import run_agent

app = FastAPI(title="Tender RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOGGER = logging.getLogger("pdf_agent.api")

class QueryPayload(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(20, ge=1, le=20, description="How many chunks to retrieve")
    source_filter: Optional[str] = Field(None, description="Optional filename to filter on")


class SourceOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    page: Optional[int]


class ToolOutput(BaseModel):
    kind: str
    format: Optional[str] = None
    note: Optional[str] = None
    content: Optional[str] = None
    confidence: Optional[int] = None
    data: Optional[Dict[str, Any]] = None


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut]


class ChatPayload(BaseModel):
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(None, description="Conversation thread id")
    top_k: int = Field(20, ge=1, le=20, description="How many chunks to retrieve")
    source_filter: Optional[str] = Field(None, description="Optional filename to filter on")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[SourceOut]
    tool_outputs: list[ToolOutput] = []


@app.on_event("startup")
def _load_env() -> None:
    # Load env if present; ignore if missing
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


@lru_cache(maxsize=1)
def _settings() -> Dict[str, Any]:
    from os import getenv

    pinecone_index = getenv("PINECONE_INDEX") or getenv("PINECONE_INDEX_NAME")
    if not pinecone_index:
        raise RuntimeError("PINECONE_INDEX (or PINECONE_INDEX_NAME) must be set")
    prefer_lots = getenv("PREFER_LOTS", "true").lower() in {"1", "true", "yes"}
    return {
        "pinecone_index": pinecone_index,
        "namespace": getenv("PINECONE_NAMESPACE") or None,
        "model": getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11"),
        "embed_model": getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
        "dimensions": int(getenv("OPENAI_EMBED_DIM", "1024")),
        "openai_api_key": getenv("OPENAI_API_TOKEN") or getenv("OPENAI_API_KEY"),
        "pinecone_api_key": getenv("PINECONE_TOKEN") or getenv("PINECONE_API_KEY"),
        "prefer_lots": prefer_lots,
    }


def _upload_max_mb() -> int:
    from os import getenv

    return int(getenv("UPLOAD_MAX_MB", "20"))


@app.post("/ask", response_model=AnswerOut)
def ask(payload: QueryPayload) -> AnswerOut:
    cfg = _settings()
    try:
        result = run_agent(
            query=payload.query,
            top_k=payload.top_k,
            source_filter=payload.source_filter,
            pinecone_index=cfg["pinecone_index"],
            namespace=cfg["namespace"],
            model=cfg["model"],
            embed_model=cfg["embed_model"],
            dimensions=cfg["dimensions"],
            openai_api_key=cfg["openai_api_key"],
            pinecone_api_key=cfg["pinecone_api_key"],
            prefer_lots=cfg["prefer_lots"],
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Agent failed in /ask")
        raise HTTPException(status_code=500, detail="Agent failed") from exc

    sources_out = []
    for src in result.get("sources", []):
        sources_out.append(
            SourceOut(
                chunk_id=src.get("chunk_id", ""),
                score=src.get("score", 0.0),
                text=src.get("text", ""),
                source=src.get("source", ""),
                page=src.get("page"),
            )
        )
    return AnswerOut(answer=result.get("answer", ""), sources=sources_out)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatPayload) -> ChatResponse:
    cfg = _settings()
    thread_id = payload.thread_id or str(uuid4())

    try:
        result = run_agent(
            query=payload.message,
            top_k=payload.top_k,
            source_filter=payload.source_filter,
            pinecone_index=cfg["pinecone_index"],
            namespace=cfg["namespace"],
            model=cfg["model"],
            embed_model=cfg["embed_model"],
            dimensions=cfg["dimensions"],
            openai_api_key=cfg["openai_api_key"],
            pinecone_api_key=cfg["pinecone_api_key"],
            prefer_lots=cfg["prefer_lots"],
            session_id=thread_id,
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Agent failed in /chat")
        raise HTTPException(status_code=500, detail="Agent failed") from exc

    sources_out = []
    for src in result.get("sources", []):
        sources_out.append(
            SourceOut(
                chunk_id=src.get("chunk_id", ""),
                score=src.get("score", 0.0),
                text=src.get("text", ""),
                source=src.get("source", ""),
                page=src.get("page"),
            )
        )

    tool_outputs = [
        ToolOutput(
            kind=item.get("kind", ""),
            format=item.get("format"),
            note=item.get("note"),
            content=item.get("content"),
            confidence=item.get("confidence"),
            data=item.get("data"),
        )
        for item in result.get("tool_outputs", [])
        if isinstance(item, dict) and item.get("kind")
    ]

    return ChatResponse(
        thread_id=thread_id,
        answer=result.get("answer", ""),
        sources=sources_out,
        tool_outputs=tool_outputs,
    )



@app.post("/chat-file", response_model=ChatResponse)
async def chat_file(
    message: str = Form(...),
    thread_id: Optional[str] = Form(None),
    top_k: int = Form(20),
    source_filter: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> ChatResponse:
    cfg = _settings()
    effective_thread_id = thread_id or str(uuid4())

    data = await file.read()
    max_bytes = _upload_max_mb() * 1024 * 1024
    if max_bytes > 0 and len(data) > max_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is too large")
    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(prefix="upload_", suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)

    try:
        result = run_agent(
            query=message,
            top_k=top_k,
            source_filter=source_filter,
            pinecone_index=cfg["pinecone_index"],
            namespace=cfg["namespace"],
            model=cfg["model"],
            embed_model=cfg["embed_model"],
            dimensions=cfg["dimensions"],
            openai_api_key=cfg["openai_api_key"],
            pinecone_api_key=cfg["pinecone_api_key"],
            prefer_lots=cfg["prefer_lots"],
            session_id=effective_thread_id,
            uploaded_file_path=str(temp_path),
            uploaded_file_name=filename,
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Agent failed in /chat-file")
        raise HTTPException(status_code=500, detail="Agent failed") from exc
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass

    sources_out = []
    for src in result.get("sources", []):
        sources_out.append(
            SourceOut(
                chunk_id=src.get("chunk_id", ""),
                score=src.get("score", 0.0),
                text=src.get("text", ""),
                source=src.get("source", ""),
                page=src.get("page"),
            )
        )

    tool_outputs = [
        ToolOutput(
            kind=item.get("kind", ""),
            format=item.get("format"),
            note=item.get("note"),
            content=item.get("content"),
            confidence=item.get("confidence"),
            data=item.get("data"),
        )
        for item in result.get("tool_outputs", [])
        if isinstance(item, dict) and item.get("kind")
    ]

    return ChatResponse(
        thread_id=effective_thread_id,
        answer=result.get("answer", ""),
        sources=sources_out,
        tool_outputs=tool_outputs,
    )

