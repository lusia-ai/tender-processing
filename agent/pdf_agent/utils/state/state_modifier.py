from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage, SystemMessage

from pdf_agent.core.state import AgentState
from pdf_agent.utils.llm.language import (
    should_chat_history_query,
    should_find_similar_query,
    should_force_tender_breakdown,
    should_readiness_query,
)
from pdf_agent.utils.retrieval.sources import build_context_block


def build_extra_system_messages(state: AgentState, query: str) -> List[BaseMessage]:
    extra: List[BaseMessage] = []
    force_breakdown = should_force_tender_breakdown(query)
    if force_breakdown:
        extra.append(
            SystemMessage(
                content=(
                    "User explicitly asked for a tender breakdown. "
                    "You MUST call tender_breakdown and use its output. "
                    "Do NOT call company_readiness unless the user explicitly asks about qualification/feasibility."
                )
            )
        )
    if not force_breakdown and should_readiness_query(query):
        extra.append(
            SystemMessage(
                content=(
                    "User is asking about qualification/feasibility/readiness. "
                    "You MUST call company_readiness and answer strictly from its output. "
                    "Do NOT answer without calling company_readiness."
                )
            )
        )
    if should_find_similar_query(query):
        extra.append(
            SystemMessage(
                content=(
                    "User is asking for similar/comparable tenders. "
                    "You MUST call find_similar_tenders and answer strictly from its output."
                )
            )
        )
    if should_chat_history_query(query):
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
        context = build_context_block(similar_sources)
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
    return extra
