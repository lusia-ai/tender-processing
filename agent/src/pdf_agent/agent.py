from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import InjectedState, create_react_agent
from pinecone import Pinecone
from pdf_agent.chunker import chunk_pdf
from pdf_agent.company_db import (
    get_company_profile,
    get_company_deliveries,
    get_tender_activity,
    get_tender_lots,
    list_tenders,
    search_lots,
)
from typing_extensions import TypedDict

try:
    import tiktoken  # type: ignore
except Exception:  # noqa: BLE001
    tiktoken = None


_CHECKPOINTER = MemorySaver()
_TOOLS_LOGGER = logging.getLogger("pdf_agent.tools")


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    is_last_step: IsLastStep
    source_filter: Optional[str]
    top_k: int
    prefer_lots: bool
    uploaded_sources: List[Dict[str, Any]]
    uploaded_file_path: Optional[str]
    uploaded_file_name: Optional[str]


def main() -> None:
    args = _build_args().parse_args()
    result = run_agent(
        query=args.query,
        pinecone_index=args.pinecone_index,
        namespace=args.namespace,
        top_k=args.top_k,
        source_filter=args.source_filter,
        prefer_lots=not args.no_prefer_lots,
        model=args.model,
        embed_model=args.embed_model,
        dimensions=args.dimensions,
        openai_api_key=args.api_key,
        pinecone_api_key=args.pinecone_api_key,
        env_file=args.env_file,
    )
    _print_result(result)


def run_agent(
    query: str,
    pinecone_index: str,
    namespace: Optional[str] = None,
    top_k: int = 6,
    source_filter: Optional[str] = None,
    prefer_lots: bool = True,
    model: str = "gpt-5.2-2025-12-11",
    embed_model: str = "text-embedding-3-large",
    dimensions: int = 1024,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    env_file: Optional[str] = None,
    session_id: Optional[str] = None,
    uploaded_sources: Optional[List[Dict[str, Any]]] = None,
    uploaded_file_path: Optional[str] = None,
    uploaded_file_name: Optional[str] = None,
) -> dict:
    if env_file:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
    _configure_logging()

    openai_api_key = openai_api_key or os.getenv("OPENAI_API_TOKEN") or os.getenv("OPENAI_API_KEY")
    pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_TOKEN") or os.getenv("PINECONE_API_KEY")
    if not openai_api_key:
        raise SystemExit("Missing OPENAI_API_TOKEN/OPENAI_API_KEY")
    if not pinecone_api_key:
        raise SystemExit("Missing PINECONE_TOKEN/PINECONE_API_KEY")

    llm = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0.0,
    )
    embedder = OpenAIEmbeddings(
        model=embed_model,
        api_key=openai_api_key,
        dimensions=dimensions,
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)

    graph = build_graph(
        llm,
        embedder,
        index,
        namespace=namespace,
        top_k=top_k,
        source_filter=source_filter,
        prefer_lots=prefer_lots,
    )
    thread_id = session_id or str(uuid4())
    state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "source_filter": source_filter,
        "top_k": top_k,
        "prefer_lots": prefer_lots,
    }
    if uploaded_sources is not None:
        state["uploaded_sources"] = uploaded_sources
    if uploaded_file_path:
        state["uploaded_file_path"] = uploaded_file_path
    if uploaded_file_name:
        state["uploaded_file_name"] = uploaded_file_name
    result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    messages = list(result.get("messages", []))
    answer = _extract_answer(messages)
    sources = _collect_sources_from_messages(messages)
    return {**result, "answer": answer, "sources": sources}


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Console RAG agent over Pinecone using LangGraph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--pinecone-index", required=True, help="Pinecone index name.")
    parser.add_argument("--namespace", help="Pinecone namespace.", default=None)
    parser.add_argument("--top-k", type=int, default=6, help="How many chunks to retrieve.")
    parser.add_argument("--source-filter", help="Restrict retrieval to a specific source filename (e.g., ted_812-2018_EN.pdf).")
    parser.add_argument("--no-prefer-lots", action="store_true", help="Disable lot-first retrieval.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model.")
    parser.add_argument("--embed-model", default="text-embedding-3-large", help="OpenAI embedding model.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="Embedding dimensions to match Pinecone index.",
    )
    parser.add_argument("--api-key", help="OpenAI API key (fallback to env).")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key (fallback to env).")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Optional .env file path.",
    )
    return parser


# Graph definition
def build_graph(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
    top_k: int,
    source_filter: Optional[str],
    prefer_lots: bool,
):
    tools = _build_tools(llm, embedder, index, namespace)
    agent = create_react_agent(
        llm,
        tools,
        state_schema=AgentState,
        state_modifier=_state_modifier,
    )
    graph = StateGraph(AgentState)
    graph.add_node("ingest_upload", ingest_upload)
    graph.add_node("agent", agent)
    graph.set_entry_point("ingest_upload")
    graph.add_edge("ingest_upload", "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=_CHECKPOINTER)


def _state_modifier(state: AgentState) -> Sequence[BaseMessage]:
    messages = list(state.get("messages", []))
    trimmed = _trim_history(messages, _max_history_tokens(), _rag_token_model())
    system = SystemMessage(content=_system_prompt(state))
    return [system, *trimmed]


def _system_prompt(state: AgentState) -> str:
    upload_names = _uploaded_file_names(state.get("uploaded_sources", []))
    upload_hint = (
        f"Uploaded tender file(s) available: {', '.join(upload_names)}."
        if upload_names
        else "No uploaded tender file is attached."
    )
    return (
        "You are a tender RAG assistant. Decide which tool to call before answering. "
        "Use tools to fetch evidence for any tender facts. "
        "If the user asks about tender content, call search_tenders or read_uploaded_tender first. "
        "If the user asks for similar tenders or feasibility (can we deliver/qualify), call find_similar_tenders. "
        "If the user asks for a chat summary, call summarize_chat. "
        "If the user asks for tender requirements only, call extract_tender_requirements. "
        "If the user asks for a tender summary, call summarize_tender. "
        "If the user asks for a structured report (JSON or table), call tender_report. "
        "If the user asks whether the company can qualify/perform, call company_readiness. "
        "If the user asks about company profile, capabilities, or tender participation status, use company DB tools. "
        "If the user asks about past deliveries or delivery history, use company_delivery_history. "
        "If a tool returns report_text, output it verbatim with no extra commentary. "
        "Use chat history only for meta questions about the conversation. "
        "Cite sources as file#page for tender facts. "
        "If evidence is missing, say you don't have enough document evidence and ask to attach the tender file. "
        "Always respond in the same language as the user.\n"
        f"{upload_hint}"
    )


def _uploaded_file_names(sources: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for src in sources:
        source = src.get("source") or ""
        if source.startswith("uploaded:"):
            names.append(source.replace("uploaded:", "", 1))
    return sorted(set(name for name in names if name))


def _normalize_tender_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.startswith("uploaded:"):
        cleaned = cleaned.replace("uploaded:", "", 1)
    if cleaned.lower().endswith(".pdf"):
        cleaned = Path(cleaned).stem
    return cleaned or None


def _infer_tender_id_from_sources(sources: List[Dict[str, Any]]) -> Optional[str]:
    for src in sources:
        source = src.get("source") or ""
        tender_id = _normalize_tender_id(source)
        if tender_id:
            return tender_id
    return None


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    _TOOLS_LOGGER.setLevel(level)


def _preview_text(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _format_tool_fields(fields: Dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_tool_start(tool_name: str, **fields: Any) -> None:
    _TOOLS_LOGGER.info("tool=%s status=start %s", tool_name, _format_tool_fields(fields))


def _log_tool_success(tool_name: str, **fields: Any) -> None:
    _TOOLS_LOGGER.info("tool=%s status=success %s", tool_name, _format_tool_fields(fields))


def _log_tool_error(tool_name: str, error: Exception) -> None:
    _TOOLS_LOGGER.exception("tool=%s status=error error=%s", tool_name, error)


def _normalize_report_format(value: Optional[str]) -> str:
    if not value:
        return "text"
    normalized = value.strip().lower()
    if normalized in {"table", "таблица", "табличный"}:
        return "table"
    if normalized in {"text", "txt", "текст", "описание"}:
        return "text"
    return "json"


def _infer_report_format(value: Optional[str], question: str) -> str:
    if value:
        return _normalize_report_format(value)
    text = (question or "").lower()
    wants_json = "json" in text or "джсон" in text or "schema" in text or "структур" in text
    wants_table = "table" in text or "табл" in text
    if wants_table:
        return "table"
    if wants_json:
        return "json"
    return "text"


def _profile_is_placeholder(profile: Dict[str, Any]) -> bool:
    if not profile:
        return True
    name = (profile.get("name") or "").strip().lower()
    notes = (profile.get("notes") or "").strip().lower()
    if not name:
        return True
    if "not specified in tender documents" in name:
        return True
    if "not present" in notes or "no company data" in notes:
        return True
    capabilities = profile.get("capabilities") or []
    if not capabilities:
        return True
    return False


def _format_company_profile(profile: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Name: {profile.get('name')}")
    if profile.get("description"):
        lines.append(f"Description: {profile.get('description')}")
    if profile.get("country"):
        lines.append(f"Country: {profile.get('country')}")
    if profile.get("website"):
        lines.append(f"Website: {profile.get('website')}")
    if profile.get("notes"):
        lines.append(f"Notes: {profile.get('notes')}")
    capabilities = profile.get("capabilities") or []
    if capabilities:
        lines.append("Capabilities:")
        for cap in capabilities:
            parts = [cap.get("capability"), cap.get("capacity"), cap.get("certification")]
            details = " | ".join([str(p) for p in parts if p])
            if details:
                lines.append(f"- {details}")
    return "\n".join(line for line in lines if line)


def _extract_json_snippet(raw: str) -> Optional[str]:
    if not raw:
        return None
    start_candidates = [raw.find("{"), raw.find("[")]
    start_candidates = [idx for idx in start_candidates if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    end_candidates = [raw.rfind("}"), raw.rfind("]")]
    end_candidates = [idx for idx in end_candidates if idx >= 0]
    if not end_candidates:
        return None
    end = max(end_candidates)
    if end <= start:
        return None
    return raw[start : end + 1]


def _parse_json_payload(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        snippet = _extract_json_snippet(raw)
        if not snippet:
            return None
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None


def _ensure_json_report(raw: str, llm: ChatOpenAI, schema_hint: str) -> Tuple[Optional[Any], str]:
    parsed = _parse_json_payload(raw)
    if parsed is not None:
        return parsed, raw
    fix_prompt = [
        SystemMessage(
            content=(
                "Fix the output to valid JSON that matches the schema. "
                "Return ONLY valid JSON, no extra text.\n"
                f"Schema:\n{schema_hint}"
            )
        ),
        HumanMessage(content=raw),
    ]
    fixed = llm.invoke(fix_prompt).content.strip()
    parsed = _parse_json_payload(fixed)
    if parsed is None:
        return None, fixed
    return parsed, fixed


def _tender_report_schema() -> str:
    return (
        "{\n"
        '  "report_type": "tender_report",\n'
        '  "scope": "string",\n'
        '  "key_requirements": [{"item": "string", "source": "file#page"}],\n'
        '  "timeline": [{"item": "string", "date": "string|null", "source": "file#page"}],\n'
        '  "evaluation_criteria": [{"criterion": "string", "weight": "string|null", "source": "file#page"}],\n'
        '  "payment_terms": [{"term": "string", "source": "file#page"}],\n'
        '  "risks": [{"risk": "string", "source": "file#page"}],\n'
        '  "sources": ["file#page"]\n'
        "}"
    )


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


def _build_context_block(sources: List[Dict[str, Any]]) -> str:
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


def _format_chat_history(messages: List[BaseMessage], max_messages: int) -> str:
    lines: List[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _message_text(msg).strip()
            if text:
                lines.append(f"User: {text}")
        elif isinstance(msg, AIMessage):
            text = _message_text(msg).strip()
            if text:
                lines.append(f"Assistant: {text}")
    if max_messages > 0:
        lines = lines[-max_messages:]
    return "\n".join(lines)


def _build_tools(
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
                _log_tool_success("search_tenders", note="empty query")
                return {"kind": "search", "query": "", "sources": [], "note": "empty query"}
            state = state or {}
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = prefer_lots if prefer_lots is not None else bool(state.get("prefer_lots", True))
            _log_tool_start(
                "search_tenders",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                prefer_lots=effective_prefer,
            )
            rewritten = _rewrite_query_text(query, list(state.get("messages", [])), llm)
            query_vector = embedder.embed_query(rewritten)
            sources = _retrieve_sources_from_index(
                query_vector=query_vector,
                index=index,
                namespace=namespace,
                top_k=effective_top_k,
                source_filter=effective_filter,
                prefer_lots=effective_prefer,
            )
            _log_tool_success("search_tenders", sources=len(sources))
            return {"kind": "search", "query": rewritten, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("search_tenders", exc)
            raise

    @tool
    def read_uploaded_tender(
        max_tokens: Optional[int] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Return chunks from the uploaded tender PDF (if present)."""
        try:
            state = state or {}
            sources = list(state.get("uploaded_sources", []))
            file_names = _uploaded_file_names(sources)
            token_limit = max_tokens if max_tokens and max_tokens > 0 else _max_upload_tokens()
            _log_tool_start("read_uploaded_tender", files=",".join(file_names) or "none", max_tokens=token_limit)
            if not sources:
                _log_tool_success("read_uploaded_tender", sources=0, note="no uploaded file")
                return {"kind": "upload", "sources": [], "note": "no uploaded file"}
            limited = _limit_sources_by_tokens(sources, token_limit, _rag_token_model())
            limited = _annotate_sources(limited, "upload")
            _log_tool_success("read_uploaded_tender", sources=len(limited))
            return {"kind": "upload", "sources": limited}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("read_uploaded_tender", exc)
            raise

    @tool
    def find_similar_tenders(
        question: Optional[str] = None,
        tender_summary: Optional[str] = None,
        max_results: Optional[int] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Find tenders similar to the uploaded tender or provided summary."""
        try:
            if not _enable_similar_retrieval():
                _log_tool_success("find_similar_tenders", note="disabled")
                return {"kind": "similar", "sources": [], "note": "similar retrieval disabled"}
            state = state or {}
            uploaded_sources = list(state.get("uploaded_sources", []))
            seed_text = ""
            if uploaded_sources:
                seed_sources = _limit_sources_by_tokens(uploaded_sources, _doc_query_tokens(), _rag_token_model())
                seed_text = "\n".join(src.get("text") or "" for src in seed_sources).strip()
            if not seed_text:
                seed_text = (tender_summary or "").strip()
            if not seed_text:
                seed_text = (question or "").strip()
            if not seed_text:
                _log_tool_success("find_similar_tenders", sources=0, note="no seed text")
                return {"kind": "similar", "sources": [], "note": "no seed text"}

            user_question = question or _latest_user_message(state.get("messages", []))
            _log_tool_start("find_similar_tenders", seed_len=len(seed_text), top_k=_similar_top_k(int(state.get("top_k") or 6)))
            queries = _build_doc_queries_from_text(seed_text, user_question, llm)
            if not queries:
                queries = [user_question or seed_text]

            min_score = _min_score()
            similar_top_k = _similar_top_k(int(state.get("top_k") or 6))
            candidates: List[Dict[str, Any]] = []
            for q in queries:
                vector = embedder.embed_query(q)
                res = _query_index(index, vector, similar_top_k, namespace, None)
                candidates.extend(_matches_to_sources(res))

            filtered = _filter_by_score(candidates, min_score)
            filtered = _dedupe_sources(filtered)
            filtered = _limit_per_source(filtered, _similar_max_per_source())
            limit_total = max_results if max_results and max_results > 0 else _similar_total()
            filtered = filtered[:limit_total]
            similar_sources = _annotate_sources(filtered, "similar")
            _log_tool_success("find_similar_tenders", sources=len(similar_sources), queries=len(queries))
            return {"kind": "similar", "queries": queries, "sources": similar_sources}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("find_similar_tenders", exc)
            raise

    @tool
    def summarize_chat(
        max_messages: Optional[int] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Summarize the recent chat conversation."""
        try:
            state = state or {}
            messages = list(state.get("messages", []))
            limit = max_messages if max_messages and max_messages > 0 else 12
            summary_input = _format_chat_history(messages, limit)
            _log_tool_start("summarize_chat", messages=limit)
            if not summary_input.strip():
                _log_tool_success("summarize_chat", note="no chat history")
                return {"kind": "chat_summary", "summary": "", "note": "no chat history"}
            prompt = [
                SystemMessage(
                    content=(
                        "Summarize the conversation in 4-6 bullet points. "
                        "Focus on user goals, key constraints, and current status. "
                        "Keep the same language as the user."
                    )
                ),
                HumanMessage(content=summary_input),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            _log_tool_success("summarize_chat", chars=len(summary))
            return {"kind": "chat_summary", "summary": summary}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("summarize_chat", exc)
            raise

    @tool
    def extract_tender_requirements(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Extract only tender requirements (qualifications, certifications, bonds, deadlines, payment terms)."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            _log_tool_start(
                "extract_tender_requirements",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )

            uploaded_sources = list(state.get("uploaded_sources", []))
            if not query and not uploaded_sources:
                _log_tool_success("tender_report", format=fmt, sources=0, note="empty_query")
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
                    return {
                        "kind": "tender_report",
                        "format": "table",
                        "report": table,
                        "report_text": table,
                        "sources": [],
                        "note": "empty_query",
                    }
                if fmt == "json":
                    report_obj = {
                        "report_type": "tender_report",
                        "scope": "not_found",
                        "key_requirements": [],
                        "timeline": [],
                        "evaluation_criteria": [],
                        "payment_terms": [],
                        "risks": [],
                        "sources": [],
                    }
                    report_text = json.dumps(report_obj)
                    return {
                        "kind": "tender_report",
                        "format": "json",
                        "report": report_obj,
                        "report_text": report_text,
                        "sources": [],
                        "note": "empty_query",
                    }
                text_report = (
                    "Tender report\n"
                    "- Scope: not found\n"
                    "- Requirements: not found\n"
                    "- Timeline: not found\n"
                    "- Evaluation criteria: not found\n"
                    "- Payment terms: not found\n"
                    "- Risks: not found"
                )
                return {
                    "kind": "tender_report",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "sources": [],
                    "note": "empty_query",
                }
            if uploaded_sources:
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                rewritten = _rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = _retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )

            context = _build_context_block(sources)
            prompt = [
                SystemMessage(
                    content=(
                        "Extract only mandatory requirements and constraints from the tender excerpts. "
                        "Include qualification criteria, certifications, prior experience thresholds, "
                        "bid bonds/guarantees, deadlines, delivery terms, and payment terms. "
                        "Do not include general scope unless it is a requirement. "
                        "Cite sources as file#page for each item. "
                        "If a requirement type is missing, say it is not found in the excerpts. "
                        "Use the same language as the user."
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            requirements = response.content.strip()
            _log_tool_success("extract_tender_requirements", sources=len(sources), chars=len(requirements))
            return {"kind": "requirements", "requirements": requirements, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("extract_tender_requirements", exc)
            raise

    @tool
    def summarize_tender(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Summarize a tender: scope, timeline, evaluation criteria, and key risks."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            _log_tool_start(
                "summarize_tender",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )

            uploaded_sources = list(state.get("uploaded_sources", []))
            if not query and not uploaded_sources:
                _log_tool_success("company_readiness", format=fmt, sources=0, note="empty_query")
                if fmt == "table":
                    table = (
                        "| Requirement | Status | Evidence | Sources |\n"
                        "| --- | --- | --- | --- |\n"
                        "| Overall readiness | unknown | Missing user request | |\n"
                    )
                    return {
                        "kind": "company_readiness",
                        "format": "table",
                        "report": table,
                        "report_text": table,
                        "sources": [],
                        "note": "empty_query",
                    }
                if fmt == "json":
                    report_obj = {
                        "report_type": "company_readiness",
                        "overall_readiness": "unknown",
                        "requirements_check": [],
                        "gaps": ["missing user request"],
                        "assumptions": [],
                        "sources": [],
                    }
                    report_text = json.dumps(report_obj)
                    return {
                        "kind": "company_readiness",
                        "format": "json",
                        "report": report_obj,
                        "report_text": report_text,
                        "sources": [],
                        "note": "empty_query",
                    }
                text_report = (
                    "Company readiness\n"
                    "- Overall readiness: unknown (missing user request)\n"
                    "- Evidence: not found"
                )
                return {
                    "kind": "company_readiness",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "sources": [],
                    "note": "empty_query",
                }
            if uploaded_sources:
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                rewritten = _rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = _retrieve_sources_from_index(
                    query_vector=vector,
                    index=index,
                    namespace=namespace,
                    top_k=effective_top_k,
                    source_filter=effective_filter,
                    prefer_lots=effective_prefer,
                )

            context = _build_context_block(sources)
            prompt = [
                SystemMessage(
                    content=(
                        "Summarize the tender using the excerpts. "
                        "Cover: scope, lots (if any), timeline/deadlines, "
                        "evaluation criteria, and key risks/constraints. "
                        "Cite sources as file#page. "
                        "Use the same language as the user."
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            _log_tool_success("summarize_tender", sources=len(sources), chars=len(summary))
            return {"kind": "tender_summary", "summary": summary, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("summarize_tender", exc)
            raise

    @tool
    def tender_report(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        report_format: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Generate a structured tender report in JSON or table format."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = _infer_report_format(report_format, query)
            _log_tool_start(
                "tender_report",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                format=fmt,
            )

            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                rewritten = _rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = _retrieve_sources_from_index(
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
                    _log_tool_success("tender_report", format="table", sources=0, note="no_sources")
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
                        "scope": "not_found",
                        "key_requirements": [],
                        "timeline": [],
                        "evaluation_criteria": [],
                        "payment_terms": [],
                        "risks": [],
                        "sources": [],
                    }
                    report_text = json.dumps(report_obj)
                    _log_tool_success("tender_report", format="json", sources=0, note="no_sources")
                    return {
                        "kind": "tender_report",
                        "format": "json",
                        "report": report_obj,
                        "report_text": report_text,
                        "sources": [],
                        "note": "no_sources",
                    }
                text_report = (
                    "Tender report\n"
                    "- Scope: not found\n"
                    "- Requirements: not found\n"
                    "- Timeline: not found\n"
                    "- Evaluation criteria: not found\n"
                    "- Payment terms: not found\n"
                    "- Risks: not found"
                )
                _log_tool_success("tender_report", format="text", sources=0, note="no_sources")
                return {
                    "kind": "tender_report",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "sources": [],
                    "note": "no_sources",
                }

            context = _build_context_block(sources)
            if fmt == "table":
                prompt = [
                    SystemMessage(
                        content=(
                            "Create a tender report as a Markdown table with columns: "
                            "Section | Details | Sources. "
                            "Include sections for scope, requirements, timeline, evaluation criteria, "
                            "payment terms, and risks. "
                            "Use file#page in the Sources column. "
                            "If a section is missing, write 'not found'. "
                            "Return ONLY the table."
                        )
                    ),
                    HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
                ]
                response = llm.invoke(prompt)
                table = response.content.strip()
                _log_tool_success("tender_report", format="table", sources=len(sources))
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
                    SystemMessage(
                        content=(
                            "Create a tender report in strict JSON. "
                            "Return ONLY valid JSON matching the schema. "
                            "Use file#page for sources. "
                            "If a field is missing, use empty arrays or 'not_found'.\n"
                            f"Schema:\n{schema}"
                        )
                    ),
                    HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
                ]
                response = llm.invoke(prompt)
                raw = response.content.strip()
                parsed, fixed = _ensure_json_report(raw, llm, schema)
                payload: Dict[str, Any] = {
                    "kind": "tender_report",
                    "format": "json",
                    "report": parsed or {},
                    "report_text": fixed,
                    "sources": sources,
                }
                if parsed is None:
                    payload["error"] = "invalid_json"
                _log_tool_success("tender_report", format="json", sources=len(sources))
                return payload

            prompt = [
                SystemMessage(
                    content=(
                        "Write a concise tender report as plain text with clear sections: "
                        "Scope, Requirements, Timeline, Evaluation criteria, Payment terms, Risks. "
                        "Use bullet points where helpful. "
                        "Cite sources as file#page inline for each factual statement. "
                        "If a section is not found, state it explicitly. "
                        "Return ONLY the report text."
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            report_text = response.content.strip()
            _log_tool_success("tender_report", format="text", sources=len(sources))
            return {
                "kind": "tender_report",
                "format": "text",
                "report": report_text,
                "report_text": report_text,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("tender_report", exc)
            raise

    @tool
    def company_readiness(
        question: Optional[str] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        report_format: Optional[str] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Compare tender requirements against company profile and return readiness report."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = _infer_report_format(report_format, query)
            _log_tool_start(
                "company_readiness",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
                format=fmt,
            )

            profile = get_company_profile()
            if _profile_is_placeholder(profile):
                _log_tool_success("company_readiness", note="missing_company_profile")
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
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                rewritten = _rewrite_query_text(query, list(state.get("messages", [])), llm)
                vector = embedder.embed_query(rewritten)
                sources = _retrieve_sources_from_index(
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
                        "| Requirement | Status | Evidence | Sources |\n"
                        "| --- | --- | --- | --- |\n"
                        "| Overall readiness | unknown | Missing tender context | |\n"
                    )
                    _log_tool_success("company_readiness", format="table", sources=0, note="no_sources")
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
                    _log_tool_success("company_readiness", format="json", sources=0, note="no_sources")
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
                _log_tool_success("company_readiness", format="text", sources=0, note="no_sources")
                return {
                    "kind": "company_readiness",
                    "format": "text",
                    "report": text_report,
                    "report_text": text_report,
                    "sources": [],
                    "note": "no_sources",
                }

            company_context = _format_company_profile(profile)
            tender_context = _build_context_block(sources)
            delivery_tender_id = _normalize_tender_id(effective_filter) or _infer_tender_id_from_sources(sources)
            deliveries = get_company_deliveries(tender_id=delivery_tender_id, limit=10)
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
            delivery_context = "\n".join(delivery_lines) if delivery_lines else "No delivery history found."
            if fmt == "table":
                prompt = [
                    SystemMessage(
                        content=(
                            "Compare tender requirements against the company profile. "
                            "Return a Markdown table with columns: Requirement | Status | Evidence | Sources. "
                            "Status must be one of: met, partial, not_met, unknown. "
                            "Sources should be file#page or 'company_db'. "
                            "If evidence is missing, use 'unknown'. "
                            "Add a first row with Requirement='Overall readiness' and Status "
                            "set to likely/uncertain/unlikely/unknown with short Evidence. "
                            "Use company delivery history when assessing experience requirements. "
                            "Return ONLY the table."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                            f"Tender context:\n{tender_context}\n\nCompany delivery history:\n{delivery_context}"
                        )
                    ),
                ]
                response = llm.invoke(prompt)
                table = response.content.strip()
                _log_tool_success("company_readiness", format="table", sources=len(sources))
                return {
                    "kind": "company_readiness",
                    "format": "table",
                    "report": table,
                    "report_text": table,
                    "sources": sources,
                }

            if fmt == "json":
                schema = _readiness_report_schema()
                prompt = [
                    SystemMessage(
                        content=(
                            "Compare tender requirements against the company profile. "
                            "Return ONLY valid JSON matching the schema. "
                            "Use file#page for tender sources and 'company_db' for company evidence. "
                            "If evidence is missing, mark status as unknown and add a gap. "
                            "Do not invent company data.\n"
                            f"Schema:\n{schema}"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                            f"Tender context:\n{tender_context}\n\nCompany delivery history:\n{delivery_context}"
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
                }
                if parsed is None:
                    payload["error"] = "invalid_json"
                _log_tool_success("company_readiness", format="json", sources=len(sources))
                return payload

            prompt = [
                SystemMessage(
                    content=(
                        "Assess whether the company can qualify and deliver. "
                        "Return a polished response with these sections in order: "
                        "Executive summary, Readiness verdict, What matches, Key gaps/risks, "
                        "What would change the verdict, Recommended next steps. "
                        "Use Markdown with bold section titles (e.g., **Executive summary**). "
                        "Use bullet lists under each section. "
                        "Emphasize 1-2 key terms per bullet with **bold**. "
                        "Keep it concise and decision-focused. "
                        "Cite sources inline for every factual claim: use file#page for tender facts and "
                        "'company_db' for company profile facts and delivery history. "
                        "If a section has no evidence, say 'not found' explicitly. "
                        "Do not include any meta notes like bullet counts or formatting instructions in the output. "
                        "Do not invent company data or tender requirements."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {query}\n\nCompany profile:\n{company_context}\n\n"
                        f"Tender context:\n{tender_context}\n\nCompany delivery history:\n{delivery_context}"
                    )
                ),
            ]
            response = llm.invoke(prompt)
            report_text = response.content.strip()
            _log_tool_success("company_readiness", format="text", sources=len(sources))
            return {
                "kind": "company_readiness",
                "format": "text",
                "report": report_text,
                "report_text": report_text,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_readiness", exc)
            raise

    @tool
    def company_profile() -> dict:
        """Return company profile and capabilities from the internal database."""
        try:
            _log_tool_start("company_profile")
            profile = get_company_profile()
            _log_tool_success("company_profile", has_profile=bool(profile))
            return {"kind": "company_profile", "profile": profile}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_profile", exc)
            raise

    @tool
    def company_tenders(limit: Optional[int] = None) -> dict:
        """List tenders recorded in the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            _log_tool_start("company_tenders", limit=effective_limit)
            tenders = list_tenders(limit=effective_limit)
            _log_tool_success("company_tenders", count=len(tenders))
            return {"kind": "company_tenders", "tenders": tenders}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_tenders", exc)
            raise

    @tool
    def company_tender_lots(tender_id: str, limit: Optional[int] = None) -> dict:
        """Fetch lots for a given tender id from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 50
            _log_tool_start("company_tender_lots", tender_id=tender_id, limit=effective_limit)
            lots = get_tender_lots(tender_id, limit=effective_limit)
            _log_tool_success("company_tender_lots", count=len(lots))
            return {"kind": "company_tender_lots", "tender_id": tender_id, "lots": lots}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_tender_lots", exc)
            raise

    @tool
    def company_lot_search(keyword: str, limit: Optional[int] = None) -> dict:
        """Search tender lots by keyword in the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            _log_tool_start("company_lot_search", keyword=_preview_text(keyword), limit=effective_limit)
            results = search_lots(keyword, limit=effective_limit)
            _log_tool_success("company_lot_search", count=len(results))
            return {"kind": "company_lot_search", "results": results}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_lot_search", exc)
            raise

    @tool
    def company_tender_activity(tender_id: Optional[str] = None, limit: Optional[int] = None) -> dict:
        """Return participation/award status from the internal database."""
        try:
            effective_limit = limit if limit and limit > 0 else 20
            _log_tool_start(
                "company_tender_activity",
                tender_id=tender_id or "all",
                limit=effective_limit,
            )
            activity = get_tender_activity(tender_id=tender_id, limit=effective_limit)
            _log_tool_success("company_tender_activity", count=len(activity))
            return {"kind": "company_tender_activity", "activity": activity}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_tender_activity", exc)
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
            normalized_tender_id = _normalize_tender_id(tender_id) if tender_id else None
            _log_tool_start(
                "company_delivery_history",
                tender_id=normalized_tender_id or "all",
                keyword=_preview_text(keyword or ""),
                limit=effective_limit,
            )
            deliveries = get_company_deliveries(
                tender_id=normalized_tender_id,
                keyword=keyword,
                limit=effective_limit,
            )
            _log_tool_success("company_delivery_history", count=len(deliveries))
            return {"kind": "company_delivery_history", "deliveries": deliveries}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_delivery_history", exc)
            raise

    return [
        search_tenders,
        read_uploaded_tender,
        find_similar_tenders,
        summarize_chat,
        extract_tender_requirements,
        summarize_tender,
        tender_report,
        company_readiness,
        company_profile,
        company_tenders,
        company_tender_lots,
        company_lot_search,
        company_tender_activity,
        company_delivery_history,
    ]


def _query_index(index, vector, top_k: int, namespace: Optional[str], flt: Optional[dict]):
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace or None,
        filter=flt,
    )


def _matches_to_sources(res) -> List[Dict[str, Any]]:
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


def _retrieve_sources_from_index(
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
    min_score = _min_score()
    max_per_source = _max_per_source()
    max_rag_tokens = _max_rag_tokens()

    if prefer_lots:
        lot_filter = _merge_filters(base_filter, {"type": {"$eq": "lot"}})
        lot_res = _query_index(index, query_vector, top_k, namespace, lot_filter)
        lot_sources_raw = _matches_to_sources(lot_res)
        lot_sources = _filter_by_score(lot_sources_raw, min_score)
    else:
        lot_sources_raw = []
        lot_sources = []

    chunk_res = _query_index(index, query_vector, top_k, namespace, base_filter)
    chunk_sources_raw = _matches_to_sources(chunk_res)
    chunk_sources = _filter_by_score(chunk_sources_raw, min_score)

    if prefer_lots and not lot_sources and not chunk_sources:
        lot_sources = lot_sources_raw
        chunk_sources = chunk_sources_raw
    elif not prefer_lots and not chunk_sources:
        chunk_sources = chunk_sources_raw

    merged = _merge_sources(
        lot_sources=lot_sources,
        chunk_sources=chunk_sources,
        top_k=top_k,
        prefer_lots=prefer_lots,
        max_per_source=max_per_source,
        max_tokens=max_rag_tokens,
        model=_rag_token_model(),
    )
    return _annotate_sources(merged, "primary")


def _merge_filters(base: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    if base and extra:
        merged = dict(base)
        merged.update(extra)
        return merged
    return base or extra


def _min_score() -> float:
    return float(os.getenv("MIN_SCORE", "0.2"))


def _max_per_source() -> int:
    return int(os.getenv("MAX_PER_SOURCE", "3"))


def _max_rag_tokens() -> int:
    return int(os.getenv("MAX_RAG_TOKENS", "1600"))


def _max_upload_tokens() -> int:
    return int(os.getenv("MAX_UPLOAD_TOKENS", "1200"))


def _upload_chunk_size() -> int:
    return int(os.getenv("UPLOAD_CHUNK_SIZE", "800"))


def _upload_chunk_overlap() -> int:
    return int(os.getenv("UPLOAD_CHUNK_OVERLAP", "120"))


def _rag_token_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11")


def _enable_similar_retrieval() -> bool:
    return os.getenv("ENABLE_SIMILAR_RETRIEVAL", "true").lower() in {"1", "true", "yes"}


def _enable_doc_query_building() -> bool:
    return os.getenv("ENABLE_DOC_QUERY_BUILDING", "true").lower() in {"1", "true", "yes"}


def _doc_query_max() -> int:
    return int(os.getenv("DOC_QUERY_MAX", "4"))


def _doc_query_tokens() -> int:
    return int(os.getenv("DOC_QUERY_TOKENS", "800"))


def _similar_top_k(base_top_k: int) -> int:
    default_k = max(6, base_top_k * 3)
    return int(os.getenv("SIMILAR_TOP_K", str(default_k)))


def _similar_max_per_source() -> int:
    return int(os.getenv("SIMILAR_MAX_PER_SOURCE", "1"))


def _similar_total() -> int:
    return int(os.getenv("SIMILAR_TOTAL", "12"))


def _filter_by_score(sources: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    if min_score <= 0:
        return sources
    return [src for src in sources if (src.get("score") or 0.0) >= min_score]


def _source_key(source: Dict[str, Any]) -> str:
    chunk_id = source.get("chunk_id") or ""
    file_id = source.get("source") or ""
    page = source.get("page") or ""
    text = source.get("text") or ""
    return f"{chunk_id}|{file_id}|{page}|{hash(text)}"


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for src in sources:
        key = _source_key(src)
        if key in seen:
            continue
        seen.add(key)
        output.append(src)
    return output


def _annotate_sources(sources: List[Dict[str, Any]], relation: str) -> List[Dict[str, Any]]:
    if not sources:
        return []
    annotated: List[Dict[str, Any]] = []
    for src in sources:
        annotated.append({**src, "relation": relation})
    return annotated


def _limit_per_source(sources: List[Dict[str, Any]], max_per_source: int) -> List[Dict[str, Any]]:
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


def _build_upload_sources(path: Path, filename: str) -> List[Dict[str, Any]]:
    chunks = chunk_pdf(path, chunk_size=_upload_chunk_size(), overlap=_upload_chunk_overlap())
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
        sources = _build_upload_sources(Path(upload_path), filename)
        sources = _limit_sources_by_tokens(sources, _max_upload_tokens(), _rag_token_model())
        return {
            **state,
            "uploaded_sources": sources,
            "uploaded_file_path": None,
            "uploaded_file_name": None,
        }
    if state.get("uploaded_sources"):
        return {**state, "uploaded_file_path": None, "uploaded_file_name": None}
    return {**state, "uploaded_sources": []}






def _limit_sources_by_tokens(
    sources: List[Dict[str, Any]],
    max_tokens: int,
    model: str,
) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        return sources
    selected: List[Dict[str, Any]] = []
    remaining = max_tokens
    for src in sources:
        tokens = _count_text_tokens(src.get("text") or "", model)
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


def _merge_sources(
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
        key = _source_key(src)
        if key in used_keys:
            return False
        file_id = src.get("source") or ""
        if per_source.get(file_id, 0) >= max_per_source:
            return False
        tokens = _count_text_tokens(src.get("text") or "", model)
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


def _enable_query_rewrite() -> bool:
    return os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in {"1", "true", "yes"}


def _rewrite_max_messages() -> int:
    return int(os.getenv("REWRITE_MAX_MESSAGES", "6"))


def _rewrite_query_text(query: str, messages: List[BaseMessage], llm: ChatOpenAI) -> str:
    if not query or not query.strip():
        return ""
    if not _enable_query_rewrite() or len(messages) < 2:
        return query
    history = messages[-_rewrite_max_messages():]
    prompt = [
        SystemMessage(
            content=(
                "Rewrite the user's latest message into a standalone search query for retrieval. "
                "Use the conversation history for context. "
                "If the user message is already specific, return it unchanged. "
                "Keep the same language as the user. "
                "Return only the rewritten query, with no extra text."
            )
        ),
        *history,
        HumanMessage(content=f"User message: {query}"),
    ]
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    return rewritten or query


def _build_doc_queries_from_text(seed_text: str, user_query: str, llm: ChatOpenAI) -> List[str]:
    if not _enable_doc_query_building():
        return []
    if not seed_text.strip():
        return []
    prompt = [
        SystemMessage(
            content=(
                "Generate 3-5 concise search queries to find similar tenders. "
                "Use only the tender content provided. Keep the language consistent with the user. "
                "Return a JSON array of strings with no extra text."
            )
        ),
        HumanMessage(content=f"User question:\n{user_query}\n\nTender content:\n{seed_text}"),
    ]
    response = llm.invoke(prompt)
    raw = response.content.strip()

    queries: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            queries = [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        lines = [line.strip("-• \t") for line in raw.splitlines()]
        queries = [line for line in lines if line]

    if not queries and user_query:
        queries = [user_query]
    max_q = max(1, _doc_query_max())
    unique: List[str] = []
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        unique.append(q)
        if len(unique) >= max_q:
            break
    return unique


def _extract_answer(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                continue
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _collect_sources_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        payload = _parse_tool_payload(msg.content)
        for src in _extract_sources_from_payload(payload):
            key = _source_key(src)
            if key in seen:
                continue
            seen.add(key)
            sources.append(src)
    return sources


def _parse_tool_payload(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


def _extract_sources_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_sources = payload.get("sources", [])
        if isinstance(raw_sources, list):
            return [src for src in raw_sources if isinstance(src, dict)]
    if isinstance(payload, list):
        return [src for src in payload if isinstance(src, dict)]
    return []



def _print_result(state: dict) -> None:
    print("=== Answer ===")
    print(state.get("answer", ""))
    print("\n=== Sources ===")
    for src in state.get("sources", []):
        label = f"{src.get('source','?')}#p{src.get('page')}" if src.get("page") else src.get("source", "?")
        print(f"- {label} (score={src.get('score'):.3f})")


def _max_history_tokens() -> int:
    return int(os.getenv("MAX_HISTORY_TOKENS", "1200"))


def _latest_user_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _count_tokens(messages_or_message, model: str) -> int:
    if isinstance(messages_or_message, BaseMessage):
        messages = [messages_or_message]
    else:
        messages = list(messages_or_message)
    text = "\n".join(_message_text(msg) for msg in messages)
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _count_text_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    return _count_tokens([HumanMessage(content=text)], model)


def _trim_history(messages: List[BaseMessage], max_tokens: int, model: str) -> List[BaseMessage]:
    if not messages:
        return []
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=lambda msgs: _count_tokens(msgs, model),
        strategy="last",
        start_on="human",
        include_system=False,
    )


if __name__ == "__main__":
    main()
