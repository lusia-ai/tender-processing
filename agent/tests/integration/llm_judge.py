from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json

from openai import OpenAI


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    score: int
    missing: List[str]
    reasons: List[str]


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return JSON: {cleaned}")
    return json.loads(cleaned[start : end + 1])


def judge_with_llm(
    *,
    api_key: str,
    model: str,
    question: str,
    answer: str,
    tool_output: Dict[str, Any],
    rubric: str,
) -> JudgeResult:
    client = OpenAI(api_key=api_key)
    payload = {
        "question": question,
        "answer": answer,
        "tool_output": tool_output,
        "rubric": rubric,
    }
    system = (
        "You are a strict evaluator. Output ONLY valid JSON with keys: "
        "pass (boolean), score (0-100), missing (array of strings), reasons (array of strings)."
    )
    user = (
        "Evaluate the answer against the rubric. "
        "Be strict and list missing elements explicitly. "
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    data = _extract_json(content)
    return JudgeResult(
        passed=bool(data.get("pass")),
        score=int(data.get("score", 0) or 0),
        missing=list(data.get("missing") or []),
        reasons=list(data.get("reasons") or []),
    )
