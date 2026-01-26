from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState

from pdf_agent.db.company_db import get_tender_metadata
from pdf_agent.core.state import AgentState, SimilarState
from pdf_agent.utils.retrieval.deliveries import collect_delivery_matches, format_delivery_matches
from pdf_agent.utils.io.logging import log_tool_error, log_tool_start, log_tool_success
from pdf_agent.utils.llm.messages import latest_user_message
from pdf_agent.utils.llm.query import build_doc_queries_from_text
from pdf_agent.utils.retrieval.retrieval import filter_by_score, matches_to_sources, query_index
from pdf_agent.utils.config.settings import enable_similar_retrieval, doc_query_tokens, min_score, rag_token_model, similar_max_per_source, similar_top_k, similar_total
from pdf_agent.utils.retrieval.sources import annotate_sources, dedupe_sources, limit_per_source, limit_sources_by_tokens
from pdf_agent.utils.retrieval.tenders import normalize_tender_id
from pdf_agent.utils.text.text import preview_text
from pdf_agent.utils.io.serialize import to_jsonable


def _group_similar_matches(
    sources: List[Dict[str, Any]],
    limit_total: int,
    max_snippets: int = 2,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for src in sources:
        key = src.get("source") or "unknown"
        grouped.setdefault(key, []).append(src)
    matches: List[Dict[str, Any]] = []
    for source, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x.get("score", 0), reverse=True)
        tender_id = normalize_tender_id(source) or source
        snippets: List[Dict[str, Any]] = []
        for item in items_sorted[:max_snippets]:
            snippets.append(
                {
                    "page": item.get("page"),
                    "score": item.get("score", 0),
                    "snippet": preview_text(item.get("text", ""), 220),
                }
            )
        match: Dict[str, Any] = {
            "tender_id": tender_id,
            "source": source,
            "top_score": items_sorted[0].get("score", 0) if items_sorted else 0,
            "snippets": snippets,
        }
        if tender_id and not source.startswith("uploaded:"):
            try:
                meta = get_tender_metadata(tender_id)
                if meta.get("primary_title"):
                    match["title"] = meta.get("primary_title")
            except Exception:
                pass
        matches.append(match)
    matches.sort(key=lambda m: m.get("top_score", 0), reverse=True)
    return matches[:limit_total]


def _run_similar_graph(
    *,
    state: SimilarState,
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
) -> SimilarState:
    def collect_seed(current: SimilarState) -> SimilarState:
        seed_text = (current.get("seed_text") or "").strip()
        uploaded_sources = list(current.get("uploaded_sources", []))
        if not seed_text and uploaded_sources:
            seed_sources = limit_sources_by_tokens(uploaded_sources, doc_query_tokens(), rag_token_model())
            seed_text = "\n".join(src.get("text") or "" for src in seed_sources).strip()
        if not seed_text:
            seed_text = (current.get("tender_summary") or "").strip()
        user_question = (current.get("user_question") or "").strip()
        if not seed_text:
            seed_text = user_question
        return {**current, "seed_text": seed_text, "user_question": user_question}

    def build_queries(current: SimilarState) -> SimilarState:
        seed_text = (current.get("seed_text") or "").strip()
        user_question = (current.get("user_question") or "").strip()
        if not seed_text:
            return {**current, "queries": []}
        queries = build_doc_queries_from_text(seed_text, user_question, llm)
        if not queries:
            queries = [user_question or seed_text]
        return {**current, "queries": queries}

    def retrieve_candidates(current: SimilarState) -> SimilarState:
        queries = current.get("queries") or []
        if not queries:
            return {**current, "candidates": []}
        similar_top = similar_top_k(int(current.get("top_k") or 6))
        candidates: List[Dict[str, Any]] = []
        for q in queries:
            vector = embedder.embed_query(q)
            res = query_index(index, vector, similar_top, namespace, None)
            candidates.extend(matches_to_sources(res))
        return {**current, "candidates": candidates}

    def build_matches(current: SimilarState) -> SimilarState:
        candidates = current.get("candidates") or []
        if not candidates:
            return {**current, "sources": [], "matches": []}
        filtered = filter_by_score(candidates, min_score())
        filtered = dedupe_sources(filtered)
        filtered = limit_per_source(filtered, similar_max_per_source())
        limit_total = current.get("max_results") or similar_total()
        filtered = filtered[:limit_total]
        similar_sources = annotate_sources(filtered, "similar")
        matches = _group_similar_matches(similar_sources, limit_total=limit_total)
        return {**current, "sources": similar_sources, "matches": matches}

    def collect_deliveries(current: SimilarState) -> SimilarState:
        seed_text = current.get("seed_text") or ""
        user_question = current.get("user_question") or ""
        deliveries = collect_delivery_matches(seed_text, user_question)
        summary = format_delivery_matches(deliveries)
        return {**current, "company_deliveries": deliveries, "company_delivery_summary": summary}

    graph = StateGraph(SimilarState)
    graph.add_node("seed_node", collect_seed)
    graph.add_node("queries_node", build_queries)
    graph.add_node("retrieve_node", retrieve_candidates)
    graph.add_node("matches_node", build_matches)
    graph.add_node("deliveries_node", collect_deliveries)
    graph.set_entry_point("seed_node")
    graph.add_edge("seed_node", "queries_node")
    graph.add_edge("queries_node", "retrieve_node")
    graph.add_edge("retrieve_node", "matches_node")
    graph.add_edge("matches_node", "deliveries_node")
    graph.add_edge("deliveries_node", END)
    return graph.compile().invoke(state)


def run_find_similar_tenders(
    *,
    state: Optional[AgentState],
    question: Optional[str],
    tender_summary: Optional[str],
    max_results: Optional[int],
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
    log_name: str,
) -> dict:
    if not enable_similar_retrieval():
        log_tool_success(log_name, note="disabled")
        return {"kind": "similar", "sources": [], "note": "similar retrieval disabled"}
    state = state or {}
    user_question = (question or latest_user_message(state.get("messages", [])) or "").strip()
    uploaded_sources = list(state.get("uploaded_sources", []))
    seed_hint = tender_summary or user_question
    log_tool_start(log_name, seed_len=len(seed_hint or ""), top_k=similar_top_k(int(state.get("top_k") or 6)))
    graph_state: SimilarState = {
        "uploaded_sources": uploaded_sources,
        "tender_summary": tender_summary or "",
        "user_question": user_question,
        "top_k": int(state.get("top_k") or 6),
        "max_results": max_results,
    }
    result_state = _run_similar_graph(
        state=graph_state,
        llm=llm,
        embedder=embedder,
        index=index,
        namespace=namespace,
    )
    seed_text = (result_state.get("seed_text") or "").strip()
    if not seed_text:
        log_tool_success(log_name, sources=0, note="no seed text")
        return {"kind": "similar", "sources": [], "note": "no seed text"}
    sources = list(result_state.get("sources", []))
    matches = list(result_state.get("matches", []))
    queries = list(result_state.get("queries", []))
    delivery_matches = list(result_state.get("company_deliveries", []))
    delivery_summary = result_state.get("company_delivery_summary", "")
    log_tool_success(
        log_name,
        sources=len(sources),
        queries=len(queries),
        deliveries=len(delivery_matches),
    )
    return {
        "kind": "similar",
        "queries": queries,
        "sources": sources,
        "matches": matches,
        "company_deliveries": delivery_matches,
        "company_delivery_summary": delivery_summary,
    }


def build_similar_tools(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
) -> List[Any]:
    @tool
    def find_similar_tenders(
        question: Optional[str] = None,
        tender_summary: Optional[str] = None,
        max_results: Optional[int] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Find tenders similar to the uploaded tender or provided summary."""
        try:
            payload = run_find_similar_tenders(
                state=state,
                question=question,
                tender_summary=tender_summary,
                max_results=max_results,
                llm=llm,
                embedder=embedder,
                index=index,
                namespace=namespace,
                log_name="find_similar_tenders",
            )
            return to_jsonable(payload)
        except Exception as exc:  # noqa: BLE001
            log_tool_error("find_similar_tenders", exc)
            raise

    return [find_similar_tenders]
