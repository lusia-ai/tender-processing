from __future__ import annotations
import argparse
import ast
from collections import Counter
from numbers import Number
import json
import re
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
    get_company_compliance,
    get_company_deliveries,
    get_tender_activity,
    get_tender_lots,
    get_tender_metadata,
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


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    is_last_step: IsLastStep
    source_filter: Optional[str]
    top_k: int
    prefer_lots: bool
    uploaded_sources: List[Dict[str, Any]]
    uploaded_file_path: Optional[str]
    uploaded_file_name: Optional[str]
    similar_sources: List[Dict[str, Any]]
    similar_queries: List[str]
class SimilarState(TypedDict, total=False):
    seed_text: str
    user_question: str
    tender_summary: str
    uploaded_sources: List[Dict[str, Any]]
    top_k: int
    max_results: Optional[int]
    queries: List[str]
    candidates: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]
    company_deliveries: List[Dict[str, Any]]
    company_delivery_summary: str
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
    recent_messages = _messages_since_last_human(messages)
    answer = _extract_answer(messages)
    sources = _collect_sources_from_messages(recent_messages)
    similar_sources = list(result.get("similar_sources", []))
    if similar_sources:
        sources = _dedupe_sources(sources + similar_sources)
    tool_outputs = _collect_tool_outputs_from_messages(recent_messages)
    if not any(output.get("kind") == "tender_breakdown" for output in tool_outputs):
        if _should_force_tender_breakdown(query) or re.search(r"(разбор тендера|tender breakdown)", answer, re.IGNORECASE):
            language = _infer_user_language(query)
            fallback = _restructure_breakdown_text(answer, language, llm)
            fallback_sections = []
            fallback_title = ""
            followup = ""
            if isinstance(fallback, dict):
                if isinstance(fallback.get("title"), str):
                    fallback_title = fallback.get("title")
                raw_sections = fallback.get("sections", {})
                if isinstance(raw_sections, dict):
                    fallback_sections = _sections_from_mapping(raw_sections, language)
                elif isinstance(raw_sections, list):
                    fallback_sections = raw_sections
                if isinstance(fallback.get("assistant_followup"), str):
                    followup = fallback.get("assistant_followup").strip()
            if not fallback_sections:
                fallback_title, fallback_sections = _sectionize_breakdown_text(answer, language)
                fallback_sections = _normalize_sections(fallback_sections, language)
            else:
                fallback_sections = _normalize_sections(fallback_sections, language)
            if fallback_sections:
                fallback_sections = _split_numbered_bullets_into_sections(fallback_sections, language)
                fallback_sections = _normalize_sections(fallback_sections, language)
            fallback_title = _normalize_section_title(fallback_title)
            if followup:
                followup_title = "Next step" if language == "English" else "Что дальше"
                fallback_sections.append({"title": followup_title, "bullets": [followup], "sources": []})
            if fallback_sections:
                tool_outputs.append(
                    {
                        "kind": "tender_breakdown",
                        "format": "json",
                        "note": "fallback_from_answer",
                        "content": answer,
                        "data": {
                            "title": fallback_title,
                            "sections": fallback_sections,
                        },
                    }
                )
    return {**result, "answer": answer, "sources": sources, "tool_outputs": tool_outputs}
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
    def auto_similar(state: AgentState) -> AgentState:
        query = _latest_user_message(state.get("messages", [])) or ""
        if not _should_find_similar_query(query):
            return {**state, "similar_sources": [], "similar_queries": []}
        try:
            result = _run_find_similar_tenders(
                state=state,
                question=query,
                tender_summary=None,
                max_results=None,
                llm=llm,
                embedder=embedder,
                index=index,
                namespace=namespace,
                log_name="auto_find_similar",
            )
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("auto_find_similar", exc)
            return state
        similar_sources = list(result.get("sources", []))
        similar_queries = list(result.get("queries", []))
        return {**state, "similar_sources": similar_sources, "similar_queries": similar_queries}
    graph = StateGraph(AgentState)
    graph.add_node("ingest_upload", ingest_upload)
    graph.add_node("auto_similar", auto_similar)
    graph.add_node("agent", agent)
    graph.set_entry_point("ingest_upload")
    graph.add_edge("ingest_upload", "auto_similar")
    graph.add_edge("auto_similar", "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=_CHECKPOINTER)
def _prune_tool_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    output: List[BaseMessage] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            j = i + 1
            tool_msgs: List[BaseMessage] = []
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_msgs.append(messages[j])
                j += 1
            if tool_msgs:
                output.append(msg)
                output.extend(tool_msgs)
            i = j
            continue
        if isinstance(msg, ToolMessage):
            i += 1
            continue
        output.append(msg)
        i += 1
    return output
def _infer_user_language(text: str) -> str:
    return "Russian" if re.search(r"[А-Яа-яЁё]", text or "") else "English"
def _state_user_language(state: dict, fallback: str = "") -> str:
    text = _latest_user_message(state.get("messages", [])) or fallback
    return _infer_user_language(text)
def _language_instruction(language: str) -> str:
    if language == "Russian":
        return (
            "Respond ONLY in Russian. Do not answer in other languages even if sources are not in Russian. "
            "Translate non-Russian terms to Russian; you may keep originals in parentheses."
        )
    return (
        "Respond ONLY in English. Do not answer in other languages even if sources are not in English. "
        "Translate non-English terms to English; you may keep originals in parentheses."
    )
def _system_prompt(state: AgentState) -> str:
    language = _state_user_language(state)
    language_instruction = _language_instruction(language)
    return (
        "You are a tender RAG assistant. Decide which tool to call before answering. "
        "Use tool outputs verbatim for structured responses. "
        "If the user even hints at qualification/feasibility/readiness (e.g., can we do it, fit, qualify, potyanem), you MUST call company_readiness and answer from its output. "
        "If the user asks for similar/comparable tenders, you MUST call find_similar_tenders and answer from its output. "
        "If the user asks about chat/history/previous messages, you MUST call summarize_chat and answer from its output. "
        "Call tender_breakdown for detailed tender parsing. "
        "Call summarize_tender for short summaries. "
        "Cite sources as file#page when available. "
        f"{language_instruction}"
    )
def _state_modifier(state: AgentState) -> Sequence[BaseMessage]:
    messages = list(state.get("messages", []))
    trimmed = _trim_history(messages, _max_history_tokens(), _rag_token_model())
    trimmed = _prune_tool_messages(trimmed)
    system = SystemMessage(content=_system_prompt(state))
    extra: List[BaseMessage] = []
    query = _latest_user_message(state.get("messages", [])) or ""
    if _should_force_tender_breakdown(query):
        extra.append(
            SystemMessage(
                content=(
                    "User explicitly asked for a tender breakdown. "
                    "You MUST call tender_breakdown and use its output. "
                    "Do NOT call company_readiness unless the user explicitly asks about qualification/feasibility."
                )
            )
        )
    if _should_readiness_query(query):
        extra.append(
            SystemMessage(
                content=(
                    "User is asking about qualification/feasibility/readiness. "
                    "You MUST call company_readiness and answer strictly from its output. "
                    "Do NOT answer without calling company_readiness."
                )
            )
        )
    if _should_find_similar_query(query):
        extra.append(
            SystemMessage(
                content=(
                    "User is asking for similar/comparable tenders. "
                    "You MUST call find_similar_tenders and answer strictly from its output."
                )
            )
        )
    if _should_chat_history_query(query):
        extra.append(
            SystemMessage(
                content=(
                    "User is asking about chat history/previous messages. "
                    "You MUST call summarize_chat and answer strictly from its output."
                )
            )
        )
    similar_sources = list(state.get("similar_sources", []))
    if similar_sources:
        context = _build_context_block(similar_sources)
        queries = [q for q in state.get("similar_queries", []) if q]
        queries_line = "Similarity queries: " + ", ".join(queries) + "." if queries else ""

        extra.append(
            SystemMessage(
                content=(
                    "You have additional similarity results from the internal database. "
                    "Use these only when answering similarity questions. "
                    f"{queries_line}\n{context}"
                )
            )
        )
    return [system, *extra, *trimmed]



def _preview_text(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _log_tool_start(tool: str, **kwargs: Any) -> None:
    details = " ".join(f"{k}={v}" for k, v in kwargs.items() if v not in (None, ""))
    message = f"tool={tool} status=start"
    if details:
        message = f"{message} {details}"
    _TOOLS_LOGGER.info(message)


def _log_tool_success(tool: str, **kwargs: Any) -> None:
    details = " ".join(f"{k}={v}" for k, v in kwargs.items() if v not in (None, ""))
    message = f"tool={tool} status=success"
    if details:
        message = f"{message} {details}"
    _TOOLS_LOGGER.info(message)


def _log_tool_error(tool: str, error: Exception) -> None:
    _TOOLS_LOGGER.error("tool=%s status=error error=%s", tool, error, exc_info=True)


def _uploaded_file_names(sources: List[Dict[str, Any]]) -> List[str]:
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


def _normalize_tender_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if raw.startswith("uploaded:"):
        raw = raw.split(":", 1)[1]
    raw = Path(raw).name
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower().startswith("ted_"):
        return "ted_" + raw[4:]
    if re.match(r"^\d{1,}-\d{4}_[A-Za-z]{2,}$", raw):
        return "ted_" + raw
    return raw


def _infer_tender_id_from_sources(sources: List[Dict[str, Any]]) -> Optional[str]:
    for src in sources:
        candidate = _normalize_tender_id(src.get("source"))
        if candidate:
            return candidate



def _infer_tender_year(tender_id: Optional[str]) -> Optional[int]:
    if not tender_id:
        return None
    match = re.search(r"-(\d{4})", tender_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_amount(value: str) -> Optional[int]:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_experience_threshold(tender_context: str) -> Optional[int]:
    if not tender_context:
        return None
    lines = tender_context.splitlines()
    candidates: list[int] = []
    for line in lines:
        lowered = line.lower()
        if "experience" in lowered or "опыт" in lowered or "deliver" in lowered or "постав" in lowered:
            match = re.search(r"([0-9][0-9\s.,]{5,})\s*pln", line, re.IGNORECASE)
            if match:
                amount = _parse_amount(match.group(1))
                if amount:
                    candidates.append(amount)
    if candidates:
        return max(candidates)
    match = re.search(r"([0-9][0-9\s.,]{5,})\s*pln", tender_context, re.IGNORECASE)
    if match:
        return _parse_amount(match.group(1))
    return None


def _delivery_total_in_years(deliveries: List[Dict[str, Any]], start_year: int, end_year: int) -> int:
    total = 0
    for item in deliveries:
        delivered_at = item.get("delivered_at")
        year = None
        if delivered_at is None:
            continue
        if hasattr(delivered_at, "year"):
            year = delivered_at.year
        else:
            try:
                year = int(str(delivered_at)[:4])
            except ValueError:
                year = None
        if year is None:
            continue
        if year < start_year or year > end_year:
            continue
        value = item.get("value")
        if isinstance(value, Number):
            total += int(value)
            continue
        if value is None:
            continue
        parsed = _parse_amount(str(value))
        if parsed:
            total += parsed
    return total



def _sum_delivery_values(deliveries: List[Dict[str, Any]]) -> int:
    total = 0
    for item in deliveries:
        value = item.get("value")
        if isinstance(value, Number):
            total += int(value)
            continue
        if value is None:
            continue
        parsed = _parse_amount(str(value))
        if parsed:
            total += parsed
    return total


def _tender_breakdown_section_titles(language: str) -> List[str]:
    if language == "Russian":
        return [
            "Заказчик и процедура",
            "Предмет закупки (что покупают)",
            "Лоты и состав поставки",
            "Сроки и график",
            "Критерии оценки и аукцион",
            "Требования к участникам (квалификация)",
            "Обеспечения и оплата",
            "Подача, язык, дедлайны, вскрытие",
            "Контакты",
            "Что не раскрыто в объявлении (SIWZ)",
        ]
    return [
        "Buyer & procedure",
        "Scope (what is being procured)",
        "Lots & deliverables",
        "Timeline & deadlines",
        "Award criteria & auction",
        "Qualification requirements",
        "Securities & payment terms",
        "Submission details",
        "Contacts",
        "Not covered in the notice (SIWZ)",
    ]


def _tender_breakdown_schema() -> str:
    example = {
        "report_type": "tender_breakdown",
        "title": "Tender breakdown title",
        "sections": {
            "Buyer & procedure": "Short bullets or sentences.",
            "Scope (what is being procured)": "Short bullets or sentences.",
        },
        "assistant_followup": "Ask how you can help next.",
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _tender_report_schema() -> str:
    example = {
        "report_type": "tender_report",
        "scope": "Short scope summary.",
        "key_requirements": [
            {"item": "Requirement", "source": "file#page"}
        ],
        "timeline": [
            {"item": "Event", "date": "YYYY-MM-DD", "source": "file#page"}
        ],
        "evaluation_criteria": [
            {"criterion": "Price", "weight": "100%", "source": "file#page"}
        ],
        "payment_terms": [
            {"term": "Payment term", "source": "file#page"}
        ],
        "risks": [
            {"risk": "Risk", "source": "file#page"}
        ],
        "sources": ["file#page"],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _ensure_json_report(raw: str, llm: ChatOpenAI, schema: str) -> Tuple[Optional[Any], str]:
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*", "", stripped).strip()
            stripped = stripped.rstrip("`").rstrip()
        return stripped
    cleaned = _strip_code_fence(raw)
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass
    prompt = [
        SystemMessage(
            content=(
                "Fix the output to valid JSON only. "
                "Return ONLY JSON matching the schema. "
                f"Schema:\n{schema}"
            )
        ),
        HumanMessage(content=raw),
    ]
    response = llm.invoke(prompt)
    fixed = _strip_code_fence(response.content.strip())
    try:
        return json.loads(fixed), fixed
    except json.JSONDecodeError:
        return None, fixed


def _sections_from_mapping(mapping: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for title, content in mapping.items():
        bullets: List[str] = []
        if isinstance(content, list):
            bullets = [str(item).strip() for item in content if str(item).strip()]
        elif isinstance(content, str):
            bullets = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            bullets = [str(content).strip()] if content is not None else []
        sections.append({"title": title, "bullets": bullets, "sources": []})
    return _normalize_sections(sections, language)


def _normalize_section_title(title: str) -> str:
    if not title:
        return ""
    cleaned = re.sub(r"^[\d\s\)\.\-]+", "", str(title)).strip()
    cleaned = cleaned.replace("**", "").replace("__", "").strip()
    return cleaned


def _normalize_sections(sections: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for sec in sections or []:
        if not isinstance(sec, dict):
            continue
        title = _normalize_section_title(str(sec.get("title", "")))
        bullets: List[str] = []
        raw_bullets = sec.get("bullets")
        if isinstance(raw_bullets, list):
            bullets = [str(item).strip() for item in raw_bullets if str(item).strip()]
        elif isinstance(sec.get("content"), str):
            bullets = [line.strip() for line in str(sec.get("content")).splitlines() if line.strip()]
        elif isinstance(sec.get("text"), str):
            bullets = [line.strip() for line in str(sec.get("text")).splitlines() if line.strip()]
        cleaned: List[str] = []
        for item in bullets:
            item = re.sub(r"^[\-•*]+\s*", "", item).strip()
            item = item.replace("**", "").replace("__", "").strip()
            if not item:
                continue
            if re.fullmatch(r"\d+", item):
                continue
            cleaned.append(item)
        sources = sec.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        normalized.append({"title": title, "bullets": cleaned, "sources": sources})
    return normalized


def _split_numbered_bullets_into_sections(sections: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for sec in sections:
        bullets = sec.get("bullets", []) if isinstance(sec, dict) else []
        if not bullets:
            output.append(sec)
            continue
        current = None
        for bullet in bullets:
            match = re.match(r"^\s*\d+[\)\.\-]\s+(.*)", bullet)
            if match:
                if current:
                    output.append(current)
                title = _normalize_section_title(match.group(1).strip())
                current = {"title": title, "bullets": [], "sources": sec.get("sources", [])}
                continue
            if current is None:
                current = {"title": sec.get("title", ""), "bullets": [], "sources": sec.get("sources", [])}
            current["bullets"].append(bullet)
        if current:
            output.append(current)
    return _normalize_sections(output, language)


def _sectionize_breakdown_text(text: str, language: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not text:
        title = "Tender breakdown" if language == "English" else "Разбор тендера"
        return title, []
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    title = ""
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines:
        if not title and re.search(r"(разбор тендера|tender breakdown)", line, re.IGNORECASE):
            title = _normalize_section_title(line)
            continue
        heading_match = re.match(r"^\d+[\)\.\-]\s+(.*)", line)
        if heading_match:
            heading = _normalize_section_title(heading_match.group(1))
            if current:
                sections.append(current)
            current = {"title": heading, "bullets": [], "sources": []}
            continue
        if line.endswith(":") and len(line.split()) < 10:
            heading = _normalize_section_title(line.rstrip(":"))
            if current:
                sections.append(current)
            current = {"title": heading, "bullets": [], "sources": []}
            continue
        if current is None:
            summary_title = "Summary" if language == "English" else "Сводка"
            current = {"title": summary_title, "bullets": [], "sources": []}
        bullet = re.sub(r"^[\-•*]+\s*", "", line).strip()
        if bullet:
            current["bullets"].append(bullet)
    if current:
        sections.append(current)
    return title or ("Tender breakdown" if language == "English" else "Разбор тендера"), _normalize_sections(sections, language)


def _render_tender_breakdown(report: Dict[str, Any], language: str) -> str:
    if not isinstance(report, dict):
        return ""
    title = report.get("title") if isinstance(report.get("title"), str) else ""
    sections = report.get("sections", [])
    if isinstance(sections, dict):
        sections = _sections_from_mapping(sections, language)
    if not isinstance(sections, list):
        sections = []
    lines: List[str] = []
    if title:
        lines.append(title)
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        sec_title = sec.get("title") or ""
        if sec_title:
            lines.append(sec_title)
        for bullet in sec.get("bullets", []) or []:
            lines.append(f"- {bullet}")
    followup = report.get("assistant_followup") if isinstance(report.get("assistant_followup"), str) else ""
    if followup:
        lines.append(followup)
    return "\n".join(lines).strip()


def _extract_delivery_keywords(tender_context: str, query: str) -> List[str]:
    text = f"{tender_context}\n{query}".lower()
    keywords = [
        "op230",
        "superheater",
        "steam",
        "pressure parts",
        "przegrzew",
        "reduktor",
        "przeklad",
        "gearbox",
        "motoreduktor",
        "boiler",
        "kotl",
    ]
    found = []
    for key in keywords:
        if key in text:
            found.append(key)
    tender_id = None
    match = re.search(r"ted_\d{1,}-\d{4}_[A-Za-z]{2,}", text)
    if match:
        tender_id = match.group(0)
        found.append(tender_id)
    return list(dict.fromkeys(found))


def _dedupe_deliveries(deliveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in deliveries:
        key = (
            item.get("title"),
            item.get("customer"),
            item.get("delivered_at"),
            item.get("value"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _collect_delivery_matches(tender_context: str, user_query: str) -> List[Dict[str, Any]]:
    deliveries: List[Dict[str, Any]] = []
    tender_id = None
    match = re.search(r"ted_\d{1,}-\d{4}_[A-Za-z]{2,}", f"{tender_context} {user_query}")
    if match:
        tender_id = match.group(0)
    if tender_id:
        deliveries.extend(get_company_deliveries(tender_id=_normalize_tender_id(tender_id), limit=10))
    for keyword in _extract_delivery_keywords(tender_context, user_query):
        deliveries.extend(get_company_deliveries(keyword=keyword, limit=6))
    return _dedupe_deliveries(deliveries)


def _format_delivery_matches(deliveries: List[Dict[str, Any]]) -> str:
    if not deliveries:
        return "No matching deliveries found."
    lines: List[str] = []
    for item in deliveries:
        parts: List[str] = []
        if item.get("delivered_at"):
            parts.append(str(item.get("delivered_at")))
        if item.get("title"):
            parts.append(str(item.get("title")))
        if item.get("customer"):
            parts.append(str(item.get("customer")))
        if item.get("value"):
            currency = item.get("currency") or ""
            parts.append(f"{item.get('value')} {currency}".strip())
        if parts:
            lines.append(" - ".join(parts))
    return "\n".join(lines)


def _format_company_profile(profile: Dict[str, Any]) -> str:
    if not profile:
        return "not found"
    lines = []
    for key in ("name", "description", "country", "website", "notes"):
        value = profile.get(key)
        if value:
            lines.append(f"{key}: {value}")
    capabilities = profile.get("capabilities", [])
    if isinstance(capabilities, list) and capabilities:
        lines.append("capabilities:")
        for cap in capabilities:
            if not isinstance(cap, dict):
                continue
            detail = " | ".join(
                [
                    str(cap.get("capability")) if cap.get("capability") else "",
                    str(cap.get("capacity")) if cap.get("capacity") else "",
                    str(cap.get("certification")) if cap.get("certification") else "",
                ]
            ).strip(" |")
            if detail:
                lines.append(f"- {detail}")
    return "\n".join(lines) if lines else "not found"


def _format_company_compliance(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "not found"
    lines = []
    for item in items:
        parts = [str(item.get("item_type")) if item.get("item_type") else ""]
        provider = item.get("provider")
        if provider:
            parts.append(f"provider={provider}")
        limit_value = item.get("limit_value")
        if limit_value:
            currency = item.get("currency") or ""
            parts.append(f"limit={limit_value} {currency}".strip())
        valid_from = item.get("valid_from")
        valid_to = item.get("valid_to")
        if valid_from or valid_to:
            parts.append(f"valid={valid_from}..{valid_to}")
        evidence = item.get("evidence")
        if evidence:
            parts.append(f"evidence={evidence}")
        line = " | ".join([p for p in parts if p])
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else "not found"


def _format_tender_metadata(meta: Dict[str, Any]) -> str:
    if not meta:
        return "not found"
    lines = []
    for key in ("tender_id", "source_file", "primary_title", "summary", "notes"):
        value = meta.get(key)
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "not found"


def _format_tender_lots(lots: List[Dict[str, Any]]) -> str:
    if not lots:
        return "not found"
    lines = []
    for lot in lots:
        parts = []
        if lot.get("lot_number") is not None:
            parts.append(f"lot {lot.get('lot_number')}")
        if lot.get("title"):
            parts.append(str(lot.get("title")))
        if lot.get("description"):
            parts.append(str(lot.get("description")))
        if lot.get("estimated_value"):
            cur = lot.get("estimated_currency") or ""
            parts.append(f"est={lot.get('estimated_value')} {cur}".strip())
        if lot.get("qualification_min_value"):
            cur = lot.get("qualification_currency") or ""
            parts.append(f"min={lot.get('qualification_min_value')} {cur}".strip())
        line = " | ".join(parts)
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else "not found"


def _format_tender_activity(activity: List[Dict[str, Any]]) -> str:
    if not activity:
        return "not found"
    lines = []
    for item in activity:
        parts = []
        if item.get("tender_id"):
            parts.append(str(item.get("tender_id")))
        if item.get("status"):
            parts.append(str(item.get("status")))
        if item.get("notes"):
            parts.append(str(item.get("notes")))
        line = " | ".join(parts)
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else "not found"


def _infer_report_format(report_format: Optional[str], query: str) -> str:
    if report_format:
        value = report_format.strip().lower()
        if value in {"json", "table", "text"}:
            return value
    lowered = (query or "").lower()
    if "json" in lowered:
        return "json"
    if "table" in lowered or "таблиц" in lowered:
        return "table"
    return "text"


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
    tender_id: Optional[str] = None,
) -> int:
    score = 40
    if profile:
        score += 15
    caps = profile.get("capabilities") if isinstance(profile, dict) else []
    if isinstance(caps, list) and caps:
        score += 10
    if compliance:
        score += 10
    if deliveries:
        score += 15
    if len(deliveries) >= 3:
        score += 5
    if not tender_context or "not found" in tender_context.lower():
        score -= 5
    threshold = _extract_experience_threshold(tender_context)
    tender_year = _infer_tender_year(tender_id) if tender_id else None
    if threshold and deliveries:
        if tender_year:
            total = _delivery_total_in_years(deliveries, tender_year - 5, tender_year)
        else:
            total = _sum_delivery_values(deliveries)
        if total < threshold:
            score = min(score, 70)
    return max(30, min(95, int(score)))


def _restructure_breakdown_text(text: str, language: str, llm: ChatOpenAI) -> Dict[str, Any]:
    if not text:
        return {}
    schema = _tender_breakdown_schema()
    titles = _tender_breakdown_section_titles(language)
    titles_line = "; ".join(titles)
    prompt = [
        SystemMessage(
            content=(
                "Rewrite the tender breakdown into JSON. "
                "Return ONLY valid JSON matching the schema. "
                "sections must be an object mapping section title to section content. "
                f"Use EXACTLY these section titles in this order as keys: {titles_line}. "
                "Do not include markdown. "
                f"Schema:\n{schema}"
            )
        ),
        HumanMessage(content=text),
    ]
    response = llm.invoke(prompt)
    parsed, _ = _ensure_json_report(response.content.strip(), llm, schema)
    return parsed if isinstance(parsed, dict) else {}


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
def _should_find_similar_query(query: str) -> bool:
    q = query.lower()
    hints = (
        "similar",
        "most similar",
        "closest",
        "nearest",
        "analogous",
        "comparable",
        "matching tenders",
        "похож",
        "схож",
        "аналог",
        "подобн",
        "поиск похож",
        "найди похож",
        "похожие тендер",
    )
    return any(hint in q for hint in hints)
def _should_readiness_query(query: str) -> bool:
    q = query.lower()
    hints = (
        "qualify",
        "eligib",
        "feasible",
        "can we",
        "are we able",
        "fit",
        "readiness",
        "готов",
        "готовность",
        "подходим",
        "потянем",
        "можем ли",
        "соответствуем",
        "вероятность",
    )
    return any(hint in q for hint in hints)



def _should_chat_history_query(query: str) -> bool:
    q = query.lower()
    hints = (
        "history",
        "chat history",
        "conversation history",
        "summarize chat",
        "summarize conversation",
        "what did we discuss",
        "previous messages",
        "summary of chat",
        "история",
        "история чата",
        "переписка",
        "что мы обсуждали",
        "суммаризируй",
        "сводка чата",
        "дай историю",
    )
    return any(hint in q for hint in hints)
def _should_force_tender_breakdown(query: str) -> bool:
    q = query.lower()
    hints = (
        "разбор",
        "разоб",
        "что в этом тендере",
        "расскажи про тендер",
        "разобрать тендер",
        "tender breakdown",
        "break down",
        "what is in this tender",
        "analyze this tender",
        "tender details",
    )
    return any(hint in q for hint in hints) and not _should_readiness_query(q)
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
        tender_id = _normalize_tender_id(source) or source
        snippets: List[Dict[str, Any]] = []
        for item in items_sorted[:max_snippets]:
            snippets.append(
                {
                    "page": item.get("page"),
                    "score": item.get("score", 0),
                    "snippet": _preview_text(item.get("text", ""), 220),
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
            seed_sources = _limit_sources_by_tokens(uploaded_sources, _doc_query_tokens(), _rag_token_model())
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
        queries = _build_doc_queries_from_text(seed_text, user_question, llm)
        if not queries:
            queries = [user_question or seed_text]
        return {**current, "queries": queries}
    def retrieve_candidates(current: SimilarState) -> SimilarState:
        queries = current.get("queries") or []
        if not queries:
            return {**current, "candidates": []}
        similar_top_k = _similar_top_k(int(current.get("top_k") or 6))
        candidates: List[Dict[str, Any]] = []
        for q in queries:
            vector = embedder.embed_query(q)
            res = _query_index(index, vector, similar_top_k, namespace, None)
            candidates.extend(_matches_to_sources(res))
        return {**current, "candidates": candidates}
    def build_matches(current: SimilarState) -> SimilarState:
        candidates = current.get("candidates") or []
        if not candidates:
            return {**current, "sources": [], "matches": []}
        filtered = _filter_by_score(candidates, _min_score())
        filtered = _dedupe_sources(filtered)
        filtered = _limit_per_source(filtered, _similar_max_per_source())
        limit_total = current.get("max_results") or _similar_total()
        filtered = filtered[:limit_total]
        similar_sources = _annotate_sources(filtered, "similar")
        matches = _group_similar_matches(similar_sources, limit_total=limit_total)
        return {**current, "sources": similar_sources, "matches": matches}
    def collect_deliveries(current: SimilarState) -> SimilarState:
        seed_text = current.get("seed_text") or ""
        user_question = current.get("user_question") or ""
        deliveries = _collect_delivery_matches(seed_text, user_question)
        summary = _format_delivery_matches(deliveries)
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
def _run_find_similar_tenders(
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
    if not _enable_similar_retrieval():
        _log_tool_success(log_name, note="disabled")
        return {"kind": "similar", "sources": [], "note": "similar retrieval disabled"}
    state = state or {}
    user_question = (question or _latest_user_message(state.get("messages", [])) or "").strip()
    uploaded_sources = list(state.get("uploaded_sources", []))
    seed_hint = tender_summary or user_question
    _log_tool_start(log_name, seed_len=len(seed_hint or ""), top_k=_similar_top_k(int(state.get("top_k") or 6)))
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
        _log_tool_success(log_name, sources=0, note="no seed text")
        return {"kind": "similar", "sources": [], "note": "no seed text"}
    sources = list(result_state.get("sources", []))
    matches = list(result_state.get("matches", []))
    queries = list(result_state.get("queries", []))
    delivery_matches = list(result_state.get("company_deliveries", []))
    delivery_summary = result_state.get("company_delivery_summary", "")
    _log_tool_success(
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
            return _run_find_similar_tenders(
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
            language = _state_user_language(state)
            language_instruction = _language_instruction(language)
            prompt = [
                SystemMessage(
                    content=(
                        "Summarize the conversation in 4-6 bullets. "
                        "Highlight key asks, decisions, and open questions. "
                        "Add next steps if they are implied. "
                        f"{language_instruction}"
                    )
                ),
                HumanMessage(content=summary_input),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            _log_tool_success("summarize_chat")
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
        """Extract qualification and delivery requirements from tender sources."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = _state_user_language(state, query)
            language_instruction = _language_instruction(language)
            _log_tool_start(
                "extract_tender_requirements",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                seed_query = query or "Extract tender qualification and delivery requirements."
                rewritten = _rewrite_query_text(seed_query, list(state.get("messages", [])), llm)
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
                _log_tool_success("extract_tender_requirements", sources=0, note="no_sources")
                return {
                    "kind": "tender_requirements",
                    "requirements": "",
                    "sources": [],
                    "note": "no_sources",
                }
            context = _build_context_block(sources)
            prompt = [
                SystemMessage(
                    content=(
                        "Extract qualification and delivery requirements relevant to assessing "
                        "whether a bidder can qualify and deliver. "
                        "Return 6-12 bullet points. "
                        "Each bullet must include a source in parentheses as file#page. "
                        "If a requirement is not found, do not invent it. "
                        f"{language_instruction}"
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            requirements = response.content.strip()
            _log_tool_success("extract_tender_requirements", sources=len(sources))
            return {
                "kind": "tender_requirements",
                "requirements": requirements,
                "sources": sources,
            }
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
        """Summarize a tender from available sources."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = _state_user_language(state, query)
            language_instruction = _language_instruction(language)
            _log_tool_start(
                "summarize_tender",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            uploaded_sources = list(state.get("uploaded_sources", []))
            if uploaded_sources:
                sources = _limit_sources_by_tokens(uploaded_sources, _max_upload_tokens(), _rag_token_model())
                sources = _annotate_sources(sources, "upload")
            else:
                seed_query = query or "Summarize the tender."
                rewritten = _rewrite_query_text(seed_query, list(state.get("messages", [])), llm)
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
                _log_tool_success("summarize_tender", sources=0, note="no_sources")
                return {"kind": "tender_summary", "summary": "", "sources": [], "note": "no_sources"}
            context = _build_context_block(sources)
            prompt = [
                SystemMessage(
                    content=(
                        "Provide a concise tender summary in 5-7 bullets. "
                        "Cover buyer, scope, key requirements, timeline, and evaluation criteria if present. "
                        "Include file#page citations where relevant. "
                        f"{language_instruction}"
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            _log_tool_success("summarize_tender", sources=len(sources))
            return {"kind": "tender_summary", "summary": summary, "sources": sources}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("summarize_tender", exc)
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
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 8)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            language = _state_user_language(state, query)
            language_instruction = _language_instruction(language)
            _log_tool_start(
                "tender_breakdown",
                query=_preview_text(query),
                top_k=effective_top_k,
                source_filter=effective_filter,
            )
            if language == "Russian":
                empty_query_text = "Нужен вопрос или загруженный тендерный файл."
                no_sources_text = "Не удалось найти источники по тендеру."
            else:
                empty_query_text = "Please provide a tender question or upload a tender file."
                no_sources_text = "No tender sources were found."
            uploaded_sources = list(state.get("uploaded_sources", []))
            if not query and not uploaded_sources:
                _log_tool_success("tender_breakdown", sources=0, note="empty_query")
                return {
                    "kind": "tender_breakdown",
                    "format": "text",
                    "report": empty_query_text,
                    "report_text": empty_query_text,
                    "sections": [],
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
            if not sources:
                _log_tool_success("tender_breakdown", sources=0, note="no_sources")
                return {
                    "kind": "tender_breakdown",
                    "format": "text",
                    "report": no_sources_text,
                    "report_text": no_sources_text,
                    "sections": [],
                    "sources": [],
                    "note": "no_sources",
                }
            tender_context = _build_context_block(sources)
            delivery_matches = _collect_delivery_matches(tender_context, query)
            delivery_summary = _format_delivery_matches(delivery_matches)
            tender_id = _normalize_tender_id(effective_filter) or _infer_tender_id_from_sources(sources)
            tender_meta = get_tender_metadata(tender_id) if tender_id else {}
            tender_lots = get_tender_lots(tender_id, limit=40) if tender_id else []
            tender_meta_context = _format_tender_metadata(tender_meta)
            tender_lots_context = _format_tender_lots(tender_lots)
            schema = _tender_breakdown_schema()
            section_titles = _tender_breakdown_section_titles(language)
            titles_line = "; ".join(section_titles)
            prompt = [
                SystemMessage(
                    content=(
                        "Create a structured tender breakdown in JSON. "
                        "Return ONLY valid JSON matching the schema. "
                        "sections must be an object mapping section title to section content (string). "
                        "Include assistant_followup as a single sentence offering next help. "
                        "Use file#page for tender evidence and 'company_db' for company deliveries. "
                        "Report ONLY on evidence in the provided context. "
                        "If a section has no evidence, write 'not found'. "
                        f"Use EXACTLY these section titles in this order as keys in sections: {titles_line}. "
                        "Do not number section titles and do not use markdown. "
                        f"{language_instruction}\n"
                        f"Schema:\n{schema}"
                    )
                ),
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
            parsed_obj: Dict[str, Any] = {}
            sections: List[Dict[str, Any]] = []
            title = ""
            if isinstance(parsed, list):
                sections = [item for item in parsed if isinstance(item, dict)]
                parsed_obj = {
                    "report_type": "tender_breakdown",
                    "title": "",
                    "sections": sections,
                    "sources": [],
                }
            elif isinstance(parsed, dict):
                parsed_obj = parsed
                raw_sections = parsed_obj.get("sections", [])
                if isinstance(raw_sections, dict):
                    sections = _sections_from_mapping(raw_sections, language)
                elif isinstance(raw_sections, list):
                    sections = raw_sections
                else:
                    sections = []
                title = parsed_obj.get("title") if isinstance(parsed_obj.get("title"), str) else ""
            raw_text = (fixed or raw or "").strip()
            if sections:
                sections = _split_numbered_bullets_into_sections(sections, language)
                sections = _normalize_sections(sections, language)
            if not sections:
                fallback_title, fallback_sections = _sectionize_breakdown_text(raw_text, language)
                fallback_sections = _normalize_sections(fallback_sections, language)
                if fallback_sections:
                    sections = fallback_sections
                    if not title:
                        title = fallback_title
            title = _normalize_section_title(title)
            followup = ""
            if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("assistant_followup"), str):
                followup = parsed_obj.get("assistant_followup").strip()
            memo_titles = {"full memo", "полный отчет", "полный отчёт", "full report", "summary", "сводка"}
            normalized_title = title.lower() if title else ""
            filtered_sections = []
            for section in sections:
                section_title = _normalize_section_title(str(section.get("title", "")))
                section_key = section_title.lower() if section_title else ""
                if section_key in memo_titles:
                    continue
                if section_title and re.search(r"(разбор тендера|tender breakdown)", section_title, re.IGNORECASE):
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
                followup_title = "Next step" if language == "English" else "Что дальше"
                sections.append({"title": followup_title, "bullets": [followup], "sources": []})
            if isinstance(parsed_obj, dict):
                parsed_obj["sections"] = sections
                if title:
                    parsed_obj["title"] = title
            report_text = _render_tender_breakdown(parsed_obj, language) if parsed_obj else (fixed or raw)
            _log_tool_success("tender_breakdown", sources=len(sources), sections=len(sections))
            return {
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
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("tender_breakdown", exc)
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
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = _infer_report_format(report_format, query)
            language = _state_user_language(state, query)
            language_instruction = _language_instruction(language)
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
                        "scope": "not found",
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
                text_report = "No tender sources found."
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
                            f"{language_instruction} "
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
                            f"{language_instruction}\n"
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
                        "Create a short executive tender brief in Markdown (not a table). "
                        "Use bold section titles: **Executive summary**, **Key requirements**, "
                        "**Commercial terms**, **Risks**, **Next steps**. "
                        "Keep each section to 2-4 bullets and avoid long paragraphs. "
                        "Keep bullets under ~160 characters when possible. "
                        "Emphasize key terms with **bold**. "
                        f"{language_instruction} "
                        "Cite sources as file#page inline for each factual claim. "
                        "Return ONLY the brief."
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
        """Compare tender requirements against company profile and return a short readiness memo."""
        try:
            state = state or {}
            query = (question or _latest_user_message(state.get("messages", []))).strip()
            effective_top_k = top_k if top_k and top_k > 0 else int(state.get("top_k") or 6)
            effective_filter = source_filter or state.get("source_filter")
            effective_prefer = bool(state.get("prefer_lots", True))
            fmt = _infer_report_format(report_format, query)
            language = _state_user_language(state, query)
            language_instruction = _language_instruction(language)
            confidence_label = "Confidence" if language == "English" else "Уверенность"
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
                history_sources = _recent_tender_sources(state.get("messages", []))
                if history_sources:
                    sources = history_sources
                else:
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
            compliance = get_company_compliance(profile_id=profile.get("id"), limit=20)
            compliance_context = _format_company_compliance(compliance)
            tender_context = _build_context_block(sources)
            delivery_tender_id = _normalize_tender_id(effective_filter) or _infer_tender_id_from_sources(sources)
            tender_meta = get_tender_metadata(delivery_tender_id) if delivery_tender_id else {}
            tender_lots = get_tender_lots(delivery_tender_id, limit=50) if delivery_tender_id else []
            tender_activity = get_tender_activity(delivery_tender_id, limit=20) if delivery_tender_id else []
            deliveries = get_company_deliveries(tender_id=delivery_tender_id, limit=10)
            keyword_deliveries: List[Dict[str, Any]] = []
            for keyword in _extract_delivery_keywords(tender_context, query):
                keyword_deliveries.extend(get_company_deliveries(keyword=keyword, limit=6))
            deliveries = _dedupe_deliveries(deliveries + keyword_deliveries)
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
            tender_year = _infer_tender_year(delivery_tender_id) if delivery_tender_id else None
            if deliveries and tender_year:
                total = _delivery_total_in_years(deliveries, tender_year - 5, tender_year)
                summary_lines.append(f"Total delivered value for last 5 years relative to tender year {tender_year}: {total} PLN")
            elif deliveries:
                total = _sum_delivery_values(deliveries)
                summary_lines.append(f"Total delivered value (all records): {total} PLN")
            summary_prefix = "\n".join(summary_lines)
            delivery_context = "\n".join(delivery_lines) if delivery_lines else "No delivery history found."
            if summary_prefix:
                delivery_context = summary_prefix + "\n" + delivery_context
            confidence = _compute_readiness_confidence(profile, compliance, deliveries, tender_context, delivery_tender_id)
            _log_tool_success(
                "company_readiness_context",
                tender_meta=bool(tender_meta),
                lots=len(tender_lots),
                activity=len(tender_activity),
                deliveries=len(deliveries),
                compliance=len(compliance),
                confidence=confidence,
            )
            tender_meta_context = _format_tender_metadata(tender_meta)
            tender_lots_context = _format_tender_lots(tender_lots)
            tender_activity_context = _format_tender_activity(tender_activity)
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
                            f"{language_instruction} "
                            "Return ONLY the table."
                        )
                    ),
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
                _log_tool_success("company_readiness", format="table", sources=len(sources))
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
                    SystemMessage(
                        content=(
                            "Compare tender requirements against the company profile. "
                            "Return ONLY valid JSON matching the schema. "
                            "Use file#page for tender sources and 'company_db' for company evidence. "
                            "If evidence is missing, mark status as unknown and add a gap. "
                            "Do not invent company data.\n"
                            f"{language_instruction}\n"
                            f"Schema:\n{schema}"
                        )
                    ),
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
                _log_tool_success("company_readiness", format="json", sources=len(sources))
                return payload
            prompt = [
                SystemMessage(
                    content=(
                        "Assess whether the company can qualify and deliver. "
                        "Write a short, decision-ready memo in Markdown (no tables). "
                        "Start with a one-line **Verdict:** high/medium/low/uncertain + one sentence why. "
                        f"Add a separate line **{confidence_label}:** {confidence}% using the provided score. "
                        "Then add a **Short summary** section (label it \"Short summary\" in English or \"Коротко\" in Russian) with 1-2 bullets focused on what is missing. "
                        "Then include **Key reasons**, **Open gaps**, and **Next steps** as short bullet lists. "
                        "Limit each section to 2-4 bullets. "
                        "Avoid long paragraphs and keep bullets to one line when possible. "
                        "Emphasize key terms with **bold**. "
                        f"{language_instruction} "
                        "Cite sources inline for every factual claim: use file#page for tender facts and "
                        "'company_db' for company profile facts and delivery history. "
                        "If evidence is missing, say 'not found' and state what you need to confirm. "
                        "Do not include meta formatting instructions in the output. "
                        "Return ONLY the memo text."
                    )
                ),
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
            _log_tool_success("company_readiness", format="text", sources=len(sources))
            return {
                "kind": "company_readiness",
                "format": "text",
                "report": report_text,
                "report_text": report_text,
                "sources": sources,
                "confidence": confidence,
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
    def company_compliance(limit: Optional[int] = None) -> dict:
        '''Return company compliance/financial readiness items from the internal database.'''
        try:
            effective_limit = limit if limit and limit > 0 else 20
            _log_tool_start("company_compliance", limit=effective_limit)
            profile = get_company_profile()
            items = get_company_compliance(profile_id=profile.get("id") if profile else None, limit=effective_limit)
            _log_tool_success("company_compliance", count=len(items))
            return {"kind": "company_compliance", "items": items}
        except Exception as exc:  # noqa: BLE001
            _log_tool_error("company_compliance", exc)
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
        tender_breakdown,
        tender_report,
        company_readiness,
        company_profile,
        company_compliance,
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
def _messages_since_last_human(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return list(messages[idx:])
    return list(messages)


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
        if not payload:
            continue
        extracted = _extract_sources_from_payload(payload) or []
        for src in extracted:
            key = _source_key(src)
            if key in seen:
                continue
            seen.add(key)
            sources.append(src)
    return sources
def _collect_tool_outputs_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        payload = _parse_tool_payload(msg.content)
        if not isinstance(payload, dict):
            continue
        kind = payload.get("kind")
        if not kind:
            continue
        content = (
            payload.get("report_text")
            or payload.get("summary")
            or payload.get("requirements")
            or ""
        )
        data = None
        if kind == "tender_breakdown":
            payload_sections = payload.get("sections", [])
            payload_title = payload.get("title")
            report_obj = payload.get("report") if isinstance(payload.get("report"), dict) else None
            if (not payload_sections or not isinstance(payload_sections, list)) and report_obj:
                payload_sections = report_obj.get("sections", [])
            if not payload_title and report_obj:
                payload_title = report_obj.get("title")
            language_hint = _infer_user_language(str(payload.get("report_text") or payload.get("content") or ""))
            if isinstance(payload_sections, dict):
                payload_sections = _sections_from_mapping(payload_sections, language_hint)
            data = {
                "title": payload_title,
                "sections": payload_sections if isinstance(payload_sections, list) else [],
                "company_deliveries": payload.get("company_deliveries", []),
                "company_delivery_summary": payload.get("company_delivery_summary", ""),
            }
        elif kind == "similar":

            data = {
                "matches": payload.get("matches", []),
                "queries": payload.get("queries", []),
                "company_deliveries": payload.get("company_deliveries", []),
                "company_delivery_summary": payload.get("company_delivery_summary", ""),
            }
        output = {
            "kind": kind,
            "format": payload.get("format"),
            "note": payload.get("note"),
            "content": content,
            "confidence": payload.get("confidence"),
            "data": data,
        }
        key = f"{kind}:{hash(content)}"
        if key in seen:
            continue
        seen.add(key)
        outputs.append(output)
    return outputs
def _parse_tool_payload(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                return None
            return value if isinstance(value, (dict, list)) else None
    return None
def _extract_sources_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_sources = payload.get("sources", [])
        if isinstance(raw_sources, list):
            return [src for src in raw_sources if isinstance(src, dict)]
        return []
    if isinstance(payload, list):
        return [src for src in payload if isinstance(src, dict)]
    return []


def _recent_tender_sources(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    tender_kinds = {"tender_breakdown", "tender_report", "summarize_tender", "extract_tender_requirements"}
    for msg in reversed(list(messages)):
        if not isinstance(msg, ToolMessage):
            continue
        payload = _parse_tool_payload(msg.content)
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") not in tender_kinds:
            continue
        sources = _extract_sources_from_payload(payload)
        if sources:
            return sources
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