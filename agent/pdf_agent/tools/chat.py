from __future__ import annotations

from typing import Annotated, Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI

from pdf_agent.core.state import AgentState
from pdf_agent.prompts.loader import render_prompt
from pdf_agent.utils.llm.language import language_instruction, state_user_language
from pdf_agent.utils.io.logging import log_tool_error, log_tool_start, log_tool_success
from pdf_agent.utils.llm.messages import format_chat_history


def build_chat_tools(llm: ChatOpenAI) -> List[Any]:
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
            summary_input = format_chat_history(messages, limit)
            log_tool_start("summarize_chat", messages=limit)
            if not summary_input.strip():
                log_tool_success("summarize_chat", note="no chat history")
                return {"kind": "chat_summary", "summary": "", "note": "no chat history"}
            language = state_user_language(state)
            prompt = [
                SystemMessage(content=render_prompt("summarize_chat.txt", language_instruction=language_instruction(language))),
                HumanMessage(content=summary_input),
            ]
            response = llm.invoke(prompt)
            summary = response.content.strip()
            log_tool_success("summarize_chat")
            return {"kind": "chat_summary", "summary": summary}
        except Exception as exc:  # noqa: BLE001
            log_tool_error("summarize_chat", exc)
            raise

    return [summarize_chat]
