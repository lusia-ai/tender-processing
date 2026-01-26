from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from pdf_agent.parsing.chunker import chunk_pdf
from pdf_agent.utils.llm.messages import count_text_tokens
from pdf_agent.utils.config.settings import max_upload_tokens, rag_token_model, upload_chunk_overlap, upload_chunk_size


def uploaded_file_names(sources: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for src in sources:
        raw = str(src.get("source") or "").strip()
        if raw.startswith("uploaded:"):
            raw = raw.split(":", 1)[1]
        raw = Path(raw).name
        if not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        names.append(raw)
    return names


def source_key(source: Dict[str, Any]) -> str:
    chunk_id = source.get("chunk_id") or ""
    file_id = source.get("source") or ""
    page = source.get("page") or ""
    text = source.get("text") or ""
    return f"{chunk_id}|{file_id}|{page}|{hash(text)}"


def dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for src in sources:
        key = source_key(src)
        if key in seen:
            continue
        seen.add(key)
        output.append(src)
    return output


def annotate_sources(sources: List[Dict[str, Any]], relation: str) -> List[Dict[str, Any]]:
    if not sources:
        return []
    annotated: List[Dict[str, Any]] = []
    for src in sources:
        annotated.append({**src, "relation": relation})
    return annotated


def limit_per_source(sources: List[Dict[str, Any]], max_per_source: int) -> List[Dict[str, Any]]:
    if max_per_source <= 0:
        return sources
    output: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for src in sources:
        name = src.get("source") or ""
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > max_per_source:
            continue
        output.append(src)
    return output


def limit_sources_by_tokens(
    sources: List[Dict[str, Any]],
    max_tokens: int,
    model: str,
) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        return sources
    selected: List[Dict[str, Any]] = []
    remaining = max_tokens
    for src in sources:
        tokens = count_text_tokens(src.get("text") or "", model)
        if selected and tokens > remaining:
            continue
        selected.append(src)
        if tokens <= remaining:
            remaining -= tokens
        else:
            remaining = 0
        if remaining <= 0:
            break
    return selected


def merge_sources(
    lot_sources: List[Dict[str, Any]],
    chunk_sources: List[Dict[str, Any]],
    top_k: int,
    prefer_lots: bool,
    max_per_source: int,
    max_tokens: int,
    model: str,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    max_per_source = max_per_source if max_per_source > 0 else 10**9
    max_tokens = max_tokens if max_tokens > 0 else 10**9
    lots = sorted(lot_sources, key=lambda s: s.get("score", 0.0), reverse=True)
    chunks = sorted(chunk_sources, key=lambda s: s.get("score", 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    used_keys: set[str] = set()
    per_source: Dict[str, int] = {}
    remaining_tokens = max_tokens

    def try_add(src: Dict[str, Any]) -> bool:
        nonlocal remaining_tokens
        key = source_key(src)
        if key in used_keys:
            return False
        file_id = src.get("source") or ""
        if per_source.get(file_id, 0) >= max_per_source:
            return False
        tokens = count_text_tokens(src.get("text") or "", model)
        if selected and tokens > remaining_tokens:
            return False
        used_keys.add(key)
        per_source[file_id] = per_source.get(file_id, 0) + 1
        if tokens <= remaining_tokens:
            remaining_tokens -= tokens
        else:
            remaining_tokens = 0
        selected.append(src)
        return True

    if prefer_lots:
        lot_target = min(len(lots), max(1, top_k // 3))
        for src in lots:
            if len(selected) >= lot_target:
                break
            try_add(src)
        for src in chunks:
            if len(selected) >= top_k:
                break
            try_add(src)
        if len(selected) < top_k:
            for src in lots:
                if len(selected) >= top_k:
                    break
                try_add(src)
    else:
        for src in chunks:
            if len(selected) >= top_k:
                break
            try_add(src)
    return selected


def build_context_block(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "NO_SOURCES"
    context_blocks = []
    for src in sources:
        src_label = f"{src.get('source','unknown')}#p{src.get('page')}" if src.get("page") else src.get("source", "unknown")
        src_type = f"type={src.get('type')}" if src.get("type") else "type=chunk"
        relation = src.get("relation")
        relation_label = f" relation={relation}" if relation else ""
        context_blocks.append(f"[{src_label} {src_type}{relation_label}]\n{src.get('text','')}")
    return "\n\n".join(context_blocks)


def build_upload_sources(path: Path, filename: str) -> List[Dict[str, Any]]:
    chunks = chunk_pdf(path, chunk_size=upload_chunk_size(), overlap=upload_chunk_overlap())
    sources: List[Dict[str, Any]] = []
    safe_name = Path(filename).name
    for chunk in chunks:
        sources.append(
            {
                "chunk_id": f"upload-{uuid4().hex}-{chunk.order}",
                "score": 1.0,
                "text": chunk.text,
                "source": f"uploaded:{safe_name}",
                "page": chunk.page,
                "pages": chunk.pages,
                "type": "upload",
            }
        )
    return sources


def ingest_upload(state: dict) -> dict:
    upload_path = state.get("uploaded_file_path")
    if upload_path:
        filename = state.get("uploaded_file_name") or Path(upload_path).name
        sources = build_upload_sources(Path(upload_path), filename)
        sources = limit_sources_by_tokens(sources, max_upload_tokens(), rag_token_model())
        return {
            **state,
            "uploaded_sources": sources,
            "uploaded_file_path": None,
            "uploaded_file_name": None,
        }
    if state.get("uploaded_sources"):
        return {**state, "uploaded_file_path": None, "uploaded_file_name": None}
    return {**state, "uploaded_sources": []}
