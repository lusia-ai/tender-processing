from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import TypedDict


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
