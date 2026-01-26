from __future__ import annotations

from typing import Any, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pdf_agent.tools.chat import build_chat_tools
from pdf_agent.tools.company import build_company_tools
from pdf_agent.tools.similar import build_similar_tools
from pdf_agent.tools.tender import build_tender_tools
from pdf_agent.tools.upload import build_upload_tools


def build_tools(
    llm: ChatOpenAI,
    embedder: OpenAIEmbeddings,
    index,
    namespace: Optional[str],
) -> List[Any]:
    tools: List[Any] = []
    tools.extend(build_tender_tools(llm, embedder, index, namespace))
    tools.extend(build_upload_tools())
    tools.extend(build_similar_tools(llm, embedder, index, namespace))
    tools.extend(build_chat_tools(llm))
    tools.extend(build_company_tools(llm, embedder, index, namespace))
    return tools
