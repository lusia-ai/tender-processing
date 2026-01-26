from __future__ import annotations

import json
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pdf_agent.prompts.loader import render_prompt
from pdf_agent.utils.config.settings import (
    doc_query_max,
    enable_doc_query_building,
    enable_query_rewrite,
    rewrite_max_messages,
)


def rewrite_query_text(query: str, messages: List[BaseMessage], llm: ChatOpenAI) -> str:
    if not query or not query.strip():
        return ""
    if not enable_query_rewrite() or len(messages) < 2:
        return query
    history = messages[-rewrite_max_messages():]
    prompt = [
        SystemMessage(content=render_prompt("query_rewrite.txt")),
        *history,
        HumanMessage(content=f"User message: {query}"),
    ]
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    return rewritten or query


def build_doc_queries_from_text(seed_text: str, user_query: str, llm: ChatOpenAI) -> List[str]:
    if not enable_doc_query_building():
        return []
    if not seed_text.strip():
        return []
    prompt = [
        SystemMessage(content=render_prompt("doc_queries.txt")),
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
    max_q = max(1, doc_query_max())
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
