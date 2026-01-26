from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import trim_messages

from pdf_agent.utils.llm.breakdown import _sections_from_mapping
from pdf_agent.utils.config.settings import max_history_tokens, rag_token_model

try:
    import tiktoken  # type: ignore
except Exception:  # noqa: BLE001
    tiktoken = None


def message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
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


def latest_user_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return message_text(msg)
    return ""


def messages_since_last_human(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return list(messages[idx:])
    return list(messages)


def format_chat_history(messages: List[BaseMessage], max_messages: int) -> str:
    lines: List[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = message_text(msg).strip()
            if text:
                lines.append(f"User: {text}")
        elif isinstance(msg, AIMessage):
            text = message_text(msg).strip()
            if text:
                lines.append(f"Assistant: {text}")
    if max_messages > 0:
        lines = lines[-max_messages:]
    return "\n".join(lines)


def count_tokens(messages_or_message, model: str) -> int:
    if isinstance(messages_or_message, BaseMessage):
        messages = [messages_or_message]
    else:
        messages = list(messages_or_message)
    text = "\n".join(message_text(msg) for msg in messages)
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_text_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    return count_tokens([HumanMessage(content=text)], model)


def trim_history(messages: List[BaseMessage], max_tokens: int, model: str) -> List[BaseMessage]:
    if not messages:
        return []
    return list(
        trim_messages(
            messages,
            max_tokens=max_tokens,
            token_counter=lambda msgs: count_tokens(msgs, model),
            strategy="last",
            start_on="human",
            include_system=False,
        )
    )


def prune_tool_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
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


def _source_key(source: Dict[str, Any]) -> str:
    chunk_id = source.get("chunk_id") or ""
    file_id = source.get("source") or ""
    page = source.get("page") or ""
    text = source.get("text") or ""
    return f"{chunk_id}|{file_id}|{page}|{hash(text)}"


def extract_answer(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                continue
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def parse_tool_payload(content):
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


def extract_sources_from_payload(payload) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_sources = payload.get("sources", [])
        if isinstance(raw_sources, list):
            return [src for src in raw_sources if isinstance(src, dict)]
        return []
    if isinstance(payload, list):
        return [src for src in payload if isinstance(src, dict)]
    return []


def collect_sources_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        payload = parse_tool_payload(msg.content)
        if not payload:
            continue
        extracted = extract_sources_from_payload(payload) or []
        for src in extracted:
            key = _source_key(src)
            if key in seen:
                continue
            seen.add(key)
            sources.append(src)
    return sources


def collect_tool_outputs_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        payload = parse_tool_payload(msg.content)
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
            language_hint = "English"
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


def recent_tender_sources(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    tender_kinds = {"tender_breakdown", "tender_report", "summarize_tender", "extract_tender_requirements"}
    for msg in reversed(list(messages)):
        if not isinstance(msg, ToolMessage):
            continue
        payload = parse_tool_payload(msg.content)
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") not in tender_kinds:
            continue
        sources = extract_sources_from_payload(payload)
        if sources:
            return sources
    return []


def trim_for_system(messages: List[BaseMessage]) -> List[BaseMessage]:
    return trim_history(messages, max_history_tokens(), rag_token_model())
