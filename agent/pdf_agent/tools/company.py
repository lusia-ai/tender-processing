from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import InjectedState

from pdf_agent.db.company_db import (
    get_company_compliance,
    get_company_deliveries,
    get_company_profile,
    get_tender_activity,
    get_tender_lots,
    get_tender_metadata,
    list_tenders,
    search_lots,
)
from pdf_agent.prompts.loader import render_prompt
from pdf_agent.core.state import AgentState
from pdf_agent.utils.llm.breakdown import _ensure_json_report
from pdf_agent.utils.retrieval.deliveries import (
    dedupe_deliveries,
    delivery_total_in_years,
    extract_delivery_keywords,
    sum_delivery_values,
)
from pdf_agent.utils.text.formatting import format_company_compliance, format_company_profile, format_tender_activity, format_tender_lots, format_tender_metadata
from pdf_agent.utils.llm.language import language_instruction, state_user_language
from pdf_agent.utils.io.logging import log_tool_error, log_tool_start, log_tool_success
from pdf_agent.utils.llm.messages import latest_user_message, recent_tender_sources
from pdf_agent.utils.llm.reporting import infer_report_format
from pdf_agent.utils.retrieval.retrieval import retrieve_sources_from_index
from pdf_agent.utils.config.settings import max_upload_tokens, rag_token_model
from pdf_agent.utils.retrieval.sources import annotate_sources, build_context_block, limit_sources_by_tokens
from pdf_agent.utils.retrieval.tenders import infer_tender_id_from_sources, infer_tender_year, normalize_tender_id
from pdf_agent.utils.text.text import preview_text
from pdf_agent.utils.llm.query import rewrite_query_text


def _profile_is_placeholder(profile: Dict[str, Any]) -> bool:
    if not profile:
        return True
    name = str(profile.get("name") or "").strip().lower()
    notes = str(profile.get("notes") or "").strip().lower()
    if not name:
        return True
    if "placeholder" in notes or "profile_missing" in notes or "do_not_use" in notes:
        return True
    return False


def _compute_readiness_confidence(
    profile: Dict[str, Any],
    compliance: List[Dict[str, Any]],
    deliveries: List[Dict[str, Any]],
    tender_context: str,
    tender_id: Optional[str],
) -> int:
    score = 45
    if profile:
        score += 10
    if compliance:
        score += 5
    if deliveries:
        score += 10
    if tender_context and "not found" not in tender_context.lower():
        score += 5
    if tender_id:
        score += 5
    return max(5, min(95, score))


def _readiness_report_schema() -> str:
    return (
        "{\n"
        '  "report_type": "company_readiness",\n'
        '  "overall_readiness": "likely|uncertain|unlikely|unknown",\n'
        '  "requirements_check": [\n'
        '    {\n'
        '      "requirement": "string",\n'
        '      "status": "met|partial|not_met|unknown",\n'
        '      "evidence": "string",\n'
        '      "tender_source": "file#page|null",\n'
        '      "company_source": "company_db|null"\n'
        "    }\n"
        "  ],\n"
        '  "gaps": ["string"],\n'
        '  "assumptions": ["string"],\n'
        '  "sources": ["file#page"]\n'
        "}"
    )


def build_company_tools(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
) -> List[Any]:
    @tool
    def company_readiness(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        report_format: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Compare tender requirements against company profile and return a short readiness memo."""
        try:
            state = state or {}
            query = (question or latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = infer_report_format(report_format, query)
            language = state_user_language(state, query)
            confidence_label = "Confidence"
            log_tool_start(
                "company_readiness",
                query=preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                format=fmt,
            )
            profile = get_company_profile()
            if _profile_is_placeholder(profile):
                log_tool_success("company_readiness", note="missing_company_profile")
                if fmt == "table":
                    table = (
                        "| Requirement | Status | Evidence | Sources |\n"
                        "| --- | --- | --- | --- |\n"
                        "| Company profile provided | unknown | Missing company profile data | company_db |\n"
                    )
                    return {
                        "kind": "company_readiness",
                        "format": "table",
                        "report": table,
                        "report_text": table,
                        "note": "company_profile_missing",
                    }
                if fmt == "json":
                    report_obj = {
                        "report_type": "company_readiness",
                        "overall_readiness": "unknown",
                        "requirements_check": [],
                        "gaps": ["company profile missing"],
                        "assumptions": ["Provide company profile data to enable readiness check."],
                        "sources": [],
                    }
                    report_text = json.dumps(report_obj)
                    return {
                        "kind": "company_readiness",
                        "format": "json",
                        "report": report_obj,
                        "report_text": report_text,
                        "note": "company_profile_missing",
                    }
                text_report = (
                    "Company readiness\n"
                    "- Overall readiness: unknown (company profile missing)\n"
                    "- Next step: add company profile data (capabilities, certifications, past deliveries)."
                )
                return {
                    "kind": "company_readiness",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "note": "company_profile_missing",
                }
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
                history_sources = recent_tender_sources(state.get("messages", []))
                if history_sources:
                    sources = history_sources
                else:
                    if fmt == "table":
                        table = (
                            "| Requirement | Status | Evidence | Sources |\n"
                            "| --- | --- | --- | --- |\n"
                            "| Overall readiness | unknown | Missing tender context | |\n"
                        )
                        log_tool_success("company_readiness", format="table", sources=0, note="no_sources")
                        return {
                            "kind": "company_readiness",
                            "format": "table",
                            "report": table,
                            "report_text": table,
                            "sources": [],
                            "note": "no_sources",
                        }
                    if fmt == "json":
                        report_obj = {
                            "report_type": "company_readiness",
                            "overall_readiness": "unknown",
                            "requirements_check": [],
                            "gaps": ["tender context missing"],
                            "assumptions": [],
                            "sources": [],
                        }
                        report_text = json.dumps(report_obj)
                        log_tool_success("company_readiness", format="json", sources=0, note="no_sources")
                        return {
                            "kind": "company_readiness",
                            "format": "json",
                            "report": report_obj,
                            "report_text": report_text,
                            "sources": [],
                            "note": "no_sources",
                        }
                    text_report = (
                        "Company readiness\n"
                        "- Overall readiness: unknown (missing tender context)\n"
                        "- Evidence: not found"
                    )
                    log_tool_success("company_readiness", format="text", sources=0, note="no_sources")
                    return {
                        "kind": "company_readiness",
                        "format": "text",
                        "report": text_report,
                        "report_text": text_report,
                        "sources": [],
                        "note": "no_sources",
                    }
            company_context = format_company_profile(profile)
            compliance = get_company_compliance(profile_id=profile.get("id"), limit=20)
            compliance_context = format_company_compliance(compliance)
            tender_context = build_context_block(sources)
            delivery_tender_id = normalize_tender_id(effective_filter) or infer_tender_id_from_sources(sources)
            tender_meta = get_tender_metadata(delivery_tender_id) if delivery_tender_id else {}
            tender_lots = get_tender_lots(delivery_tender_id, limit=50) if delivery_tender_id else []
            tender_activity = get_tender_activity(delivery_tender_id, limit=20) if delivery_tender_id else []
            deliveries = get_company_deliveries(tender_id=delivery_tender_id, limit=10)
            keyword_deliveries: List[Dict[str, Any]] = []
            for keyword in extract_delivery_keywords(tender_context, query):
                keyword_deliveries.extend(get_company_deliveries(keyword=keyword, limit=6))
            deliveries = dedupe_deliveries(deliveries + keyword_deliveries)
            delivery_lines: List[str] = []
            for item in deliveries:
                parts: List[str] = []
                if item.get("delivered_at"):
                    parts.append(str(item.get("delivered_at")))
                if item.get("title"):
                    parts.append(str(item.get("title")))
                if item.get("customer"):
                    parts.append(f"customer={item.get('customer')}")
                if item.get("value"):
                    currency = item.get("currency") or ""
                    parts.append(f"value={item.get('value')} {currency}".strip())
                if item.get("scope"):
                    parts.append(f"scope={item.get('scope')}")
                if item.get("evidence"):
                    parts.append(f"evidence={item.get('evidence')}")
                if parts:
                    delivery_lines.append(f"- {' | '.join(parts)}")
            summary_lines = []
            tender_year = infer_tender_year(delivery_tender_id) if delivery_tender_id else None
            if deliveries and tender_year:
                total = delivery_total_in_years(deliveries, tender_year - 5, tender_year)
                summary_lines.append(f"Total delivered value for last 5 years relative to tender year {tender_year}: {total} PLN")
            elif deliveries:
                total = sum_delivery_values(deliveries)
                summary_lines.append(f"Total delivered value (all records): {total} PLN")
            summary_prefix = "\n".join(summary_lines)
            delivery_context = "\n".join(delivery_lines) if delivery_lines else "No delivery history found."
            if summary_prefix:
                delivery_context = summary_prefix + "\n" + delivery_context
            confidence = _compute_readiness_confidence(profile, compliance, deliveries, tender_context, delivery_tender_id)
            log_tool_success(
                "company_readiness_context",
                tender_meta=bool(tender_meta),
                lots=len(tender_lots),
                activity=len(tender_activity),
                deliveries=len(deliveries),
                compliance=len(compliance),
                confidence=confidence,
            )
            tender_meta_context = format_tender_metadata(tender_meta)
            tender_lots_context = format_tender_lots(tender_lots)
            tender_activity_context = format_tender_activity(tender_activity)
            if fmt == "table":
                prompt = [
                    SystemMessage(content=render_prompt("readiness_table.txt", language_instruction=language_instruction(language))),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                            f"Company compliance/financial readiness (DB):\n{compliance_context}\n\n"
                            f"Tender metadata (DB):\n{tender_meta_context}\n\n"
                            f"Tender lots (DB):\n{tender_lots_context}\n\n"
                            f"Tender activity (DB):\n{tender_activity_context}\n\n"
                            f"Tender context (PDF):\n{tender_context}\n\n"
                            f"Company delivery history (DB):\n{delivery_context}\n\n"
                            f"Confidence score (use as-is): {confidence}%"
                        )
                    ),
                ]
                response = llm.invoke(prompt)
                table = response.content.strip()
                log_tool_success("company_readiness", format="table", sources=len(sources))
                return {
                    "kind": "company_readiness",
                    "format": "table",
                    "report": table,
                    "report_text": table,
                    "sources": sources,
                    "confidence": confidence,
                }
            if fmt == "json":
                schema = _readiness_report_schema()
                prompt = [
                    SystemMessage(content=render_prompt("readiness_json.txt", language_instruction=language_instruction(language), schema=schema)),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                            f"Company compliance/financial readiness (DB):\n{compliance_context}\n\n"
                            f"Tender metadata (DB):\n{tender_meta_context}\n\n"
                            f"Tender lots (DB):\n{tender_lots_context}\n\n"
                            f"Tender activity (DB):\n{tender_activity_context}\n\n"
                            f"Tender context (PDF):\n{tender_context}\n\n"
                            f"Company delivery history (DB):\n{delivery_context}\n\n"
                            f"Confidence score (use as-is): {confidence}%"
                        )
                    ),
                ]
                response = llm.invoke(prompt)
                raw = response.content.strip()
                parsed, fixed = _ensure_json_report(raw, llm, schema)
                payload = {
                    "kind": "company_readiness",
                    "format": "json",
                    "report": parsed or {},
                    "report_text": fixed,
                    "sources": sources,
                    "confidence": confidence,
                }
                if parsed is None:
                    payload["error"] = "invalid_json"
                log_tool_success("company_readiness", format="json", sources=len(sources))
                return payload
            prompt = [
                SystemMessage(content=render_prompt("readiness_text.txt", language_instruction=language_instruction(language), confidence_label=confidence_label, confidence=confidence)),
                HumanMessage(
                    content=(
                        f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                        f"Company compliance/financial readiness (DB):\n{compliance_context}\n\n"
                        f"Tender metadata (DB):\n{tender_meta_context}\n\n"
                        f"Tender lots (DB):\n{tender_lots_context}\n\n"
                        f"Tender activity (DB):\n{tender_activity_context}\n\n"
                        f"Tender context (PDF):\n{tender_context}\n\n"
                        f"Company delivery history (DB):\n{delivery_context}\n\n"
                        f"Confidence score (use as-is): {confidence}%"
                    )
                ),
            ]
            response = llm.invoke(prompt)
            report_text = response.content.strip()
            log_tool_success("company_readiness", format="text", sources=len(sources))
            return {
                "kind": "company_readiness",
                "format": "text",
                "report": report_text,
                "report_text": report_text,
                "sources": sources,
                "confidence": confidence,
            }
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_readiness", exc)
            raise

    @tool
    def company_profile() -> dict:
        """Return company profile and capabilities from the internal database."""
        try:
            log_tool_start("company_profile")
            profile = get_company_profile()
            log_tool_success("company_profile", has_profile=bool(profile))
            return {"kind": "company_profile", "profile": profile}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_profile", exc)
            raise

    @tool
    def company_compliance(limit: Optional[int] = None) -> dict:
        """Return company compliance/financial readiness items from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            log_tool_start("company_compliance", limit=effective_limit)
            profile = get_company_profile()
            items = get_company_compliance(profile_id=profile.get("id") if profile else None, limit=effective_limit)
            log_tool_success("company_compliance", count=len(items))
            return {"kind": "company_compliance", "items": items}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_compliance", exc)
            raise

    @tool
    def company_tenders(limit: Optional[int] = None) -> dict:
        """List tenders recorded in the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            log_tool_start("company_tenders", limit=effective_limit)
            tenders = list_tenders(limit=effective_limit)
            log_tool_success("company_tenders", count=len(tenders))
            return {"kind": "company_tenders", "tenders": tenders}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_tenders", exc)
            raise

    @tool
    def company_tender_lots(tender_id: str, limit: Optional[int] = None) -> dict:
        """Fetch lots for a given tender id from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 50
            log_tool_start("company_tender_lots", tender_id=tender_id, limit=effective_limit)
            lots = get_tender_lots(tender_id, limit=effective_limit)
            log_tool_success("company_tender_lots", count=len(lots))
            return {"kind": "company_tender_lots", "tender_id": tender_id, "lots": lots}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_tender_lots", exc)
            raise

    @tool
    def company_lot_search(keyword: str, limit: Optional[int] = None) -> dict:
        """Search tender lots by keyword in the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            log_tool_start("company_lot_search", keyword=preview_text(keyword), limit=effective_limit)
            results = search_lots(keyword, limit=effective_limit)
            log_tool_success("company_lot_search", count=len(results))
            return {"kind": "company_lot_search", "results": results}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_lot_search", exc)
            raise

    @tool
    def company_tender_activity(tender_id: Optional[str] = None, limit: Optional[int] = None) -> dict:
        """Return participation/award status from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            log_tool_start(
                "company_tender_activity",
                tender_id=tender_id or "all",
                limit=effective_limit,
            )
            activity = get_tender_activity(tender_id=tender_id, limit=effective_limit)
            log_tool_success("company_tender_activity", count=len(activity))
            return {"kind": "company_tender_activity", "activity": activity}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_tender_activity", exc)
            raise

    @tool
    def company_delivery_history(
        tender_id: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Return past delivery history from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            normalized_tender_id = normalize_tender_id(tender_id) if tender_id else None
            log_tool_start(
                "company_delivery_history",
                tender_id=normalized_tender_id or "all",
                keyword=preview_text(keyword or ""),
                limit=effective_limit,
            )
            deliveries = get_company_deliveries(
                tender_id=normalized_tender_id,
                keyword=keyword,
                limit=effective_limit,
            )
            log_tool_success("company_delivery_history", count=len(deliveries))
            return {"kind": "company_delivery_history", "deliveries": deliveries}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("company_delivery_history", exc)
            raise

    return [
        company_readiness,
        company_profile,
        company_compliance,
        company_tenders,
        company_tender_lots,
        company_lot_search,
        company_tender_activity,
        company_delivery_history,
    ]
