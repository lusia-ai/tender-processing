from __future__ import annotations

from typing import Any, Dict, List, Optional

from pdf_agent.utils.config.settings import max_per_source, max_rag_tokens, min_score, rag_token_model
from pdf_agent.utils.retrieval.sources import annotate_sources, merge_sources


def query_index(index, vector, top_k: int, namespace: Optional[str], flt: Optional[dict]):
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace or None,
        filter=flt,
    )


def matches_to_sources(res) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for match in res.matches:
        meta = match.metadata or {}
        page = meta.get("page") or meta.get("metadata", {}).get("page") if meta else None
        pages = meta.get("pages") or meta.get("metadata", {}).get("pages") if meta else None
        if page is None and pages:
            page = pages[0]
        sources.append(
            {
                "chunk_id": match.id,
                "score": match.score,
                "text": meta.get("text") or meta.get("metadata", {}).get("text") if meta else "",
                "source": meta.get("source") or meta.get("metadata", {}).get("source") if meta else "",
                "page": page,
                "pages": pages,
                "type": meta.get("type") or meta.get("metadata", {}).get("type") if meta else None,
            }
        )
    return sources


def merge_filters(base: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    if base and extra:
        merged = dict(base)
        merged.update(extra)
        return merged
    return base or extra


def filter_by_score(sources: List[Dict[str, Any]], min_score_value: float) -> List[Dict[str, Any]]:
    if min_score_value <= 0:
        return sources
    return [src for src in sources if (src.get("score") or 0.0) >= min_score_value]


def retrieve_sources_from_index(
    query_vector,
    index,
    namespace: Optional[str],
    top_k: int,
    source_filter: Optional[str],
    prefer_lots: bool,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    base_filter = {"source": {"$eq": source_filter}} if source_filter else None
    min_score_value = min_score()
    max_source = max_per_source()
    max_tokens = max_rag_tokens()
    if prefer_lots:
        lot_filter = merge_filters(base_filter, {"type": {"$eq": "lot"}})
        lot_res = query_index(index, query_vector, top_k, namespace, lot_filter)
        lot_sources_raw = matches_to_sources(lot_res)
        lot_sources = filter_by_score(lot_sources_raw, min_score_value)
    else:
        lot_sources_raw = []
        lot_sources = []
    chunk_res = query_index(index, query_vector, top_k, namespace, base_filter)
    chunk_sources_raw = matches_to_sources(chunk_res)
    chunk_sources = filter_by_score(chunk_sources_raw, min_score_value)
    if prefer_lots and not lot_sources and not chunk_sources:
        lot_sources = lot_sources_raw
        chunk_sources = chunk_sources_raw
    elif not prefer_lots and not chunk_sources:
        chunk_sources = chunk_sources_raw
    merged = merge_sources(
        lot_sources=lot_sources,
        chunk_sources=chunk_sources,
        top_k=top_k,
        prefer_lots=prefer_lots,
        max_per_source=max_source,
        max_tokens=max_tokens,
        model=rag_token_model(),
    )
    return annotate_sources(merged, "primary")
