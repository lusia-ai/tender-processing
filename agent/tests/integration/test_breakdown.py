from __future__ import annotations

from tests.integration.api_client import chat_with_file, find_tool_output
from tests.integration.llm_judge import judge_with_llm
from tests.integration.rubrics import BREAKDOWN_RUBRIC


def test_breakdown_with_ted_5791(api_client, integration_config):
    question = (
        "Please break down the attached tender in detail "
        "(buyer, scope, lots, deadlines, qualification requirements, securities, payment terms, submission details)."
    )
    payload = chat_with_file(
        client=api_client,
        message=question,
        pdf_path=integration_config.tender_pdf_path,
        top_k=8,
    )
    tool = find_tool_output(payload, "tender_breakdown")
    assert tool is not None, "Expected tender_breakdown tool output, but none was returned"
    assert tool.get("note") not in {"empty_query", "no_sources"}, f"Breakdown failed: {tool.get('note')}"

    sections = (tool.get("data") or {}).get("sections", [])
    assert sections, "Breakdown sections are empty"

    result = judge_with_llm(
        api_key=integration_config.openai_api_key,
        model=integration_config.judge_model,
        question=question,
        answer=payload.get("answer", ""),
        tool_output=tool,
        rubric=BREAKDOWN_RUBRIC,
    )
    assert result.passed, f"LLM judge failed: score={result.score} missing={result.missing} reasons={result.reasons}"
