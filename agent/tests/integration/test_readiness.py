from __future__ import annotations

from tests.integration.api_client import chat_with_file, find_tool_output
from tests.integration.llm_judge import judge_with_llm
from tests.integration.rubrics import READINESS_RUBRIC


def test_company_readiness_with_ted_5791(api_client, integration_config):
    question = (
        "Based on the attached tender PDF, can we qualify and deliver? "
        "Give a readiness verdict (high/medium/low) with key gaps and evidence."
    )
    payload = chat_with_file(
        client=api_client,
        message=question,
        pdf_path=integration_config.tender_pdf_path,
        top_k=6,
    )
    tool = find_tool_output(payload, "company_readiness")
    assert tool is not None, "Expected company_readiness tool output, but none was returned"
    assert tool.get("note") not in {"company_profile_missing", "no_sources"}, f"Readiness failed: {tool.get('note')}"

    result = judge_with_llm(
        api_key=integration_config.openai_api_key,
        model=integration_config.judge_model,
        question=question,
        answer=payload.get("answer", ""),
        tool_output=tool,
        rubric=READINESS_RUBRIC,
    )
    assert result.passed, f"LLM judge failed: score={result.score} missing={result.missing} reasons={result.reasons}"
