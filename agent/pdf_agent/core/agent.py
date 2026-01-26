from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone

from pdf_agent.prompts.loader import render_prompt
from pdf_agent.core.state import AgentState
from pdf_agent.tools import build_tools
from pdf_agent.tools.similar import run_find_similar_tenders
from pdf_agent.utils.llm.language import (
    language_instruction,
    should_find_similar_query,
    state_user_language,
)
from pdf_agent.utils.io.logging import configure_logging, log_tool_error
from pdf_agent.utils.llm.messages import (
    collect_sources_from_messages,
    collect_tool_outputs_from_messages,
    extract_answer,
    latest_user_message,
    messages_since_last_human,
    prune_tool_messages,
    trim_history,
)
from pdf_agent.utils.config.settings import (
    default_embed_dim,
    default_embed_model,
    default_model,
    enable_auto_similar,
    max_history_tokens,
    rag_token_model,
)
from pdf_agent.utils.retrieval.sources import dedupe_sources, ingest_upload
from pdf_agent.utils.state.state_modifier import build_extra_system_messages

def run_agent(
    query: str,
    pinecone_index: str,
    namespace: Optional[str] = None,
    top_k: int = 6,
    source_filter: Optional[str] = None,
    prefer_lots: bool = True,
    model: Optional[str] = None,
    embed_model: Optional[str] = None,
    dimensions: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    env_file: Optional[str] = None,
    session_id: Optional[str] = None,
    uploaded_sources: Optional[List[Dict[str, Any]]] = None,
    uploaded_file_path: Optional[str] = None,
    uploaded_file_name: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    auto_similar: Optional[bool] = None,
) -> dict:
    if env_file:
        from pdf_agent.utils.io.env import load_env

        load_env(env_file)
    configure_logging()
    model = model or default_model()
    embed_model = embed_model or default_embed_model()
    dimensions = dimensions or default_embed_dim()
    openai_api_key = openai_api_key or _env("OPENAI_API_TOKEN", "OPENAI_API_KEY")
    pinecone_api_key = pinecone_api_key or _env("PINECONE_TOKEN", "PINECONE_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_TOKEN/OPENAI_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_TOKEN/PINECONE_API_KEY")
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
        checkpointer=checkpointer,
        auto_similar=auto_similar,
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
    recent_messages = messages_since_last_human(messages)
    answer = extract_answer(messages)
    sources = collect_sources_from_messages(recent_messages)
    similar_sources = list(result.get("similar_sources", []))
    if similar_sources:
        sources = dedupe_sources(sources + similar_sources)
    tool_outputs = collect_tool_outputs_from_messages(recent_messages)
    return {**result, "answer": answer, "sources": sources, "tool_outputs": tool_outputs}


def build_graph(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
    top_k: int,
    source_filter: Optional[str],
    prefer_lots: bool,
    checkpointer: Optional[Any] = None,
    auto_similar: Optional[bool] = None,
):
    tools = build_tools(llm, embedder, index, namespace)
    agent = create_react_agent(
        llm,
        tools,
        state_schema=AgentState,
        state_modifier=_state_modifier,
    )

    def auto_similar_node(state: AgentState) -> AgentState:
        query = latest_user_message(state.get("messages", [])) or ""
        if not should_find_similar_query(query):
            return {**state, "similar_sources": [], "similar_queries": []}
        try:
            result = run_find_similar_tenders(
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
            log_tool_error("auto_find_similar", exc)
            return state
        similar_sources = list(result.get("sources", []))
        similar_queries = list(result.get("queries", []))
        return {**state, "similar_sources": similar_sources, "similar_queries": similar_queries}

    graph = StateGraph(AgentState)
    graph.add_node("ingest_upload", ingest_upload)
    graph.add_node("agent", agent)
    graph.set_entry_point("ingest_upload")
    if auto_similar is None:
        auto_similar = enable_auto_similar()
    if auto_similar:
        graph.add_node("auto_similar", auto_similar_node)
        graph.add_edge("ingest_upload", "auto_similar")
        graph.add_edge("auto_similar", "agent")
    else:
        graph.add_edge("ingest_upload", "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


def _system_prompt(state: AgentState) -> str:
    language = state_user_language(state)
    return render_prompt("agent_system.txt", language_instruction=language_instruction(language))


def _state_modifier(state: AgentState) -> Sequence[BaseMessage]:
    messages = list(state.get("messages", []))
    trimmed = trim_history(messages, max_history_tokens(), rag_token_model())
    trimmed = prune_tool_messages(trimmed)
    system = SystemMessage(content=_system_prompt(state))
    query = latest_user_message(state.get("messages", [])) or ""
    extra = build_extra_system_messages(state, query)
    return [system, *extra, *trimmed]


def _env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


if __name__ == "__main__":
    main()
