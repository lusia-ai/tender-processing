from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import InjectedState

from pdf_agent.db.company_db import get_tender_lots, get_tender_metadata
from pdf_agent.prompts.loader import render_prompt
from pdf_agent.core.state import AgentState
from pdf_agent.utils.llm.breakdown import (
    _ensure_json_report,
    _normalize_section_title,
    _normalize_sections,
    _render_tender_breakdown,
    _sections_from_mapping,
    _split_numbered_bullets_into_sections,
    _tender_breakdown_schema,
    _tender_breakdown_section_titles,
    _tender_report_schema,
)
from pdf_agent.utils.retrieval.deliveries import collect_delivery_matches, format_delivery_matches
from pdf_agent.utils.text.formatting import format_tender_lots, format_tender_metadata
from pdf_agent.utils.llm.language import language_instruction, state_user_language
from pdf_agent.utils.io.logging import log_tool_error, log_tool_start, log_tool_success
from pdf_agent.utils.llm.messages import latest_user_message
from pdf_agent.utils.llm.query import rewrite_query_text
from pdf_agent.utils.llm.reporting import infer_report_format
from pdf_agent.utils.retrieval.retrieval import retrieve_sources_from_index
from pdf_agent.utils.llm.schemas import _model_dump, parse_tender_breakdown_report
from pdf_agent.utils.config.settings import max_upload_tokens, rag_token_model
from pdf_agent.utils.retrieval.sources import annotate_sources, build_context_block, limit_sources_by_tokens
from pdf_agent.utils.retrieval.tenders import infer_tender_id_from_sources, normalize_tender_id
from pdf_agent.utils.text.text import preview_text
from pdf_agent.utils.io.serialize import to_jsonable


def build_tender_tools(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
) -> List[Any]:
    @tool
    def search_tenders(
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        prefer_lots: Optional[bool] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Search tender chunks in the vector index. Returns sources with metadata."""
        try:
            if not query or not query.strip():
                log_tool_success("search_tenders", note="empty query")
                return {"kind": "search", "query": "", "sources": [], "note": "empty query"}
            state = state or {}
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = prefer_lots if prefer_lots is not None else bool(state.get("prefer_lots", True))
            log_tool_start(
                "search_tenders",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                prefer_lots=effective_prefer,
            )
            rewritten = rewrite_query_text(query, list(state.get("messages", [])), llm)
            query_vector = embedder.embed_query(rewritten)
            sources = retrieve_sources_from_index(
                query_vector=query_vector,
                index=index,
                namespace=namespace,
                top_k=effective_top_k,
                source_filter=effective_filter,
                prefer_lots=effective_prefer,
            )
            log_tool_success("search_tenders", sources=len(sources))
            return {"kind": "search", "query": rewritten, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("search_tenders", exc)
            raise

    @tool
    def extract_tender_requirements(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Extract qualification and delivery requirements from tender sources."""
        try:
            state = state or {}
            query = (question or latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = state_user_language(state, query)
            log_tool_start(
                "extract_tender_requirements",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = limit_sources_by_tokens(uploaded_sources, max_upload_tokens(), rag_token_model())
                sources = annotate_sources(sources, "upload")
            else:
                seed_query = query or "Extract tender qualification and delivery requirements."
                rewritten = rewrite_query_text(seed_query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )
            if not sources:
                log_tool_success("extract_tender_requirements", sources=0, note="no_sources")
                return {
                    "kind": "tender_requirements",
                    "requirements": "",
                    "sources": [],
                    "note": "no_sources",
                }
            context = build_context_block(sources)
            prompt = [
                SystemMessage(content=render_prompt("extract_requirements.txt", language_instruction=language_instruction(language))),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            requirements = response.content.strip()
            log_tool_success("extract_tender_requirements", sources=len(sources))
            return {
                "kind": "tender_requirements",
                "requirements": requirements,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            log_tool_error("extract_tender_requirements", exc)
            raise

    @tool
    def summarize_tender(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Summarize a tender from available sources."""
        try:
            state = state or {}
            query = (question or latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = state_user_language(state, query)
            log_tool_start(
                "summarize_tender",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = limit_sources_by_tokens(uploaded_sources, max_upload_tokens(), rag_token_model())
                sources = annotate_sources(sources, "upload")
            else:
                seed_query = query or "Summarize the tender."
                rewritten = rewrite_query_text(seed_query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )
            if not sources:
                log_tool_success("summarize_tender", sources=0, note="no_sources")
                return {"kind": "tender_summary", "summary": "", "sources": [], "note": "no_sources"}
            context = build_context_block(sources)
            prompt = [
                SystemMessage(content=render_prompt("summarize_tender.txt", language_instruction=language_instruction(language))),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            log_tool_success("summarize_tender", sources=len(sources))
            return {"kind": "tender_summary", "summary": summary, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("summarize_tender", exc)
            raise

    @tool
    def tender_breakdown(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Generate a structured tender breakdown with collapsible sections."""
        try:
            state = state or {}
            query = (question or latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 8)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = state_user_language(state, query)
            log_tool_start(
                "tender_breakdown",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            empty_query_text = "Please provide a tender question or upload a tender file."
            no_sources_text = "No tender sources were found."
            uploaded_sources = list(state.get("uploaded_sources", []))
            if not query and not uploaded_sources:
                log_tool_success("tender_breakdown", sources=0, note="empty_query")
                payload = {
                    "kind": "tender_breakdown",
                    "format": "text",
                    "report": empty_query_text,
                    "report_text": empty_query_text,
                    "sections": [],
                    "sources": [],
                    "note": "empty_query",
                }
                return to_jsonable(payload)
            if uploaded_sources:
                sources = limit_sources_by_tokens(uploaded_sources, max_upload_tokens(), rag_token_model())
                sources = annotate_sources(sources, "upload")
            else:
                rewritten = rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )
            if not sources:
                log_tool_success("tender_breakdown", sources=0, note="no_sources")
                payload = {
                    "kind": "tender_breakdown",
                    "format": "text",
                    "report": no_sources_text,
                    "report_text": no_sources_text,
                    "sections": [],
                    "sources": [],
                    "note": "no_sources",
                }
                return to_jsonable(payload)
            tender_context = build_context_block(sources)
            delivery_matches = collect_delivery_matches(tender_context, query)
            delivery_summary = format_delivery_matches(delivery_matches)
            tender_id = normalize_tender_id(effective_filter) or infer_tender_id_from_sources(sources)
            tender_meta = get_tender_metadata(tender_id) if tender_id else {}
            tender_lots = get_tender_lots(tender_id, limit=40) if tender_id else []
            tender_meta_context = format_tender_metadata(tender_meta)
            tender_lots_context = format_tender_lots(tender_lots)
            schema = _tender_breakdown_schema()
            section_titles = _tender_breakdown_section_titles(language)
            titles_line = "; ".join(section_titles)
            prompt = [
                SystemMessage(content=render_prompt("tender_breakdown.txt", titles_line=titles_line, language_instruction=language_instruction(language), schema=schema)),
                HumanMessage(
                    content=(
                        f"Question: {query}\n\n"
                        f"Tender metadata (DB):\n{tender_meta_context}\n\n"
                        f"Tender lots (DB):\n{tender_lots_context}\n\n"
                        f"Tender context (PDF):\n{tender_context}\n\n"
                        f"Company delivery matches (DB):\n{delivery_summary}"
                    )
                ),
            ]
            response = llm.invoke(prompt)
            raw = response.content.strip()
            parsed, fixed = _ensure_json_report(raw, llm, schema)
            if parsed is None:
                raise ValueError("tender_breakdown returned invalid JSON")
            report = parse_tender_breakdown_report(parsed)
            parsed_obj = _model_dump(report)
            sections: List[Dict[str, Any]] = []
            title = report.title or ""
            raw_sections = report.sections
            if isinstance(raw_sections, dict):
                sections = _sections_from_mapping(raw_sections, language)
            elif isinstance(raw_sections, list):
                for sec in raw_sections:
                    if hasattr(sec, "model_dump"):
                        sec = sec.model_dump()
                    elif hasattr(sec, "dict"):
                        sec = sec.dict()
                    if isinstance(sec, dict):
                        sections.append(sec)
            if sections:
                sections = _split_numbered_bullets_into_sections(sections, language)
                sections = _normalize_sections(sections, language)
            title = _normalize_section_title(title)
            followup = report.assistant_followup or ""
            memo_titles = {"full memo", "full report", "summary"}
            normalized_title = title.lower() if title else ""
            filtered_sections = []
            for section in sections:
                section_title = _normalize_section_title(str(section.get("title", "")))
                section_key = section_title.lower() if section_title else ""
                if section_key in memo_titles:
                    continue
                if section_title and "tender breakdown" in section_title.lower():
                    continue
                if normalized_title and section_key == normalized_title:
                    continue
                filtered_sections.append(section)
            sections = filtered_sections
            if section_titles:
                expected = [_normalize_section_title(title).lower() for title in section_titles]
                ordered = []
                remainder = []
                used = set()
                for sec in sections:
                    sec_key = _normalize_section_title(str(sec.get("title", ""))).lower()
                    if not sec_key:
                        remainder.append(sec)
                        continue
                    if sec_key in used:
                        continue
                    used.add(sec_key)
                    remainder.append(sec)
                for key in expected:
                    for sec in remainder:
                        sec_key = _normalize_section_title(str(sec.get("title", ""))).lower()
                        if sec_key == key and sec not in ordered:
                            ordered.append(sec)
                for sec in remainder:
                    if sec not in ordered:
                        ordered.append(sec)
                sections = ordered
            if followup:
                followup_title = "Next step"
                sections.append({"title": followup_title, "bullets": [followup], "sources": []})
            if isinstance(parsed_obj, dict):
                parsed_obj["sections"] = sections
                if title:
                    parsed_obj["title"] = title
            report_text = _render_tender_breakdown(parsed_obj, language) if parsed_obj else (fixed or raw)
            log_tool_success("tender_breakdown", sources=len(sources), sections=len(sections))
            payload = {
                "kind": "tender_breakdown",
                "format": "json",
                "report": parsed_obj or {},
                "report_text": report_text,
                "sections": sections,
                "title": title or None,
                "sources": sources,
                "company_deliveries": delivery_matches,
                "company_delivery_summary": delivery_summary,
            }
            return to_jsonable(payload)
        except Exception as exc:  # noqa: BLE001
            log_tool_error("tender_breakdown", exc)
            raise

    @tool
    def tender_report(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        report_format: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Generate a tender report (default short text; JSON/table only if explicitly requested)."""
        try:
            state = state or {}
            query = (question or latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = infer_report_format(report_format, query)
            language = state_user_language(state, query)
            log_tool_start(
                "tender_report",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                format=fmt,
            )
            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = limit_sources_by_tokens(uploaded_sources, max_upload_tokens(), rag_token_model())
                sources = annotate_sources(sources, "upload")
            else:
                rewritten = rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )
            if not sources:
                if fmt == "table":
                    table = (
                        "| Section | Details | Sources |\n"
                        "| --- | --- | --- |\n"
                        "| scope | not found | |\n"
                        "| requirements | not found | |\n"
                        "| timeline | not found | |\n"
                        "| evaluation criteria | not found | |\n"
                        "| payment terms | not found | |\n"
                        "| risks | not found | |\n"
                    )
                    log_tool_success("tender_report", format="table", sources=0, note="no_sources")
                    return {
                        "kind": "tender_report",
                        "format": "table",
                        "report": table,
                        "report_text": table,
                        "sources": [],
                        "note": "no_sources",
                    }
                if fmt == "json":
                    report_obj = {
                        "report_type": "tender_report",
                        "scope": "not found",
                        "key_requirements": [],
                        "timeline": [],
                        "evaluation_criteria": [],
                        "payment_terms": [],
                        "risks": [],
                        "sources": [],
                    }
                    report_text = json.dumps(report_obj)
                    log_tool_success("tender_report", format="json", sources=0, note="no_sources")
                    return {
                        "kind": "tender_report",
                        "format": "json",
                        "report": report_obj,
                        "report_text": report_text,
                        "sources": [],
                        "note": "no_sources",
                    }
                text_report = "No tender sources were found."
                log_tool_success("tender_report", format="text", sources=0, note="no_sources")
                return {
                    "kind": "tender_report",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "sources": [],
                    "note": "no_sources",
                }
            context = build_context_block(sources)
            if fmt == "table":
                prompt = [
                    SystemMessage(content=render_prompt("tender_report_table.txt", language_instruction=language_instruction(language))),
                    HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
                ]
                response = llm.invoke(prompt)
                table = response.content.strip()
                log_tool_success("tender_report", format="table", sources=len(sources))
                return {
                    "kind": "tender_report",
                    "format": "table",
                    "report": table,
                    "report_text": table,
                    "sources": sources,
                }
            if fmt == "json":
                schema = _tender_report_schema()
                prompt = [
                    SystemMessage(content=render_prompt("tender_report_json.txt", language_instruction=language_instruction(language), schema=schema)),
                    HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
                ]
                response = llm.invoke(prompt)
                raw = response.content.strip()
                parsed, fixed = _ensure_json_report(raw, llm, schema)
                payload = {
                    "kind": "tender_report",
                    "format": "json",
                    "report": parsed or {},
                    "report_text": fixed,
                    "sources": sources,
                }
                if parsed is None:
                    payload["error"] = "invalid_json"
                log_tool_success("tender_report", format="json", sources=len(sources))
                return payload
            prompt = [
                SystemMessage(content=render_prompt("tender_report_text.txt", language_instruction=language_instruction(language))),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            report_text = response.content.strip()
            log_tool_success("tender_report", format="text", sources=len(sources))
            return {
                "kind": "tender_report",
                "format": "text",
                "report": report_text,
                "report_text": report_text,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            log_tool_error("tender_report", exc)
            raise

    return [
        search_tenders,
        extract_tender_requirements,
        summarize_tender,
        tender_breakdown,
        tender_report,
    ]
