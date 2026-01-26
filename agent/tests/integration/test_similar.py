from __future__ import annotations

from tests.integration.api_client import chat_with_file, find_tool_output
from tests.integration.llm_judge import judge_with_llm
from tests.integration.rubrics import SIMILAR_RUBRIC


def test_similar_tenders_with_ted_5791(api_client, integration_config):
    question = "Find the 3-5 most similar tenders to the attached tender and explain why each is similar."
    payload = chat_with_file(
        client=api_client,
        message=question,
        pdf_path=integration_config.tender_pdf_path,
        top_k=6,
    )
    tool = find_tool_output(payload, "similar")
    assert tool is not None, "Expected similar tool output, but none was returned"
    assert tool.get("note") not in {"similar retrieval disabled", "no seed text"}, f"Similar retrieval failed: {tool.get('note')}"

    matches = (tool.get("data") or {}).get("matches", [])
    assert 3 <= len(matches) <= 5, f"Expected 3-5 similar tenders, got {len(matches)}"

    result = judge_with_llm(
        api_key=integration_config.openai_api_key,
        model=integration_config.judge_model,
        question=question,
        answer=payload.get("answer", ""),
        tool_output=tool,
        rubric=SIMILAR_RUBRIC,
    )
    assert result.passed, f"LLM judge failed: score={result.score} missing={result.missing} reasons={result.reasons}"
