from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from pdf_agent.core.state import AgentState
from pdf_agent.utils.io.logging import log_tool_error, log_tool_start, log_tool_success
from pdf_agent.utils.config.settings import max_upload_tokens, rag_token_model
from pdf_agent.utils.retrieval.sources import annotate_sources, limit_sources_by_tokens, uploaded_file_names


def build_upload_tools() -> List[Any]:
    @tool
    def read_uploaded_tender(
        max_tokens: Optional[int] = None,
        state: Annotated[AgentState, InjectedState] = None,
    ) -> dict:
        """Return chunks from the uploaded tender PDF (if present)."""
        try:
            state = state or {}
            sources = list(state.get("uploaded_sources", []))
            file_names = uploaded_file_names(sources)
            token_limit = max_tokens if max_tokens and max_tokens > 0 else max_upload_tokens()
            log_tool_start("read_uploaded_tender", files=",".join(file_names) or "none", max_tokens=token_limit)
            if not sources:
                log_tool_success("read_uploaded_tender", sources=0, note="no uploaded file")
                return {"kind": "upload", "sources": [], "note": "no uploaded file"}
            limited = limit_sources_by_tokens(sources, token_limit, rag_token_model())
            limited = annotate_sources(limited, "upload")
            log_tool_success("read_uploaded_tender", sources=len(limited))
            return {"kind": "upload", "sources": limited}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("read_uploaded_tender", exc)
            raise

    return [read_uploaded_tender]
