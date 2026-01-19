import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pdf_agent.response_policy import ActionType, ResponsePolicyConfig, evaluate_policy


def test_answer_plus_followup_for_clear_short_query():
    config = ResponsePolicyConfig(
        enable_proactive_followups=True,
        max_followup_items=2,
        confidence_threshold_clarify=0.3,
        length_threshold_summary=900,
    )
    decision = evaluate_policy(
        user_message="Summarize tender ted_812-2018_EN.pdf",
        history=[],
        confidence=0.8,
        estimated_answer_length=200,
        source_filter="ted_812-2018_EN.pdf",
        config=config,
    )
    assert decision.action_type == ActionType.ANSWER_PLUS_FOLLOWUP
    assert decision.followups


def test_ask_clarify_on_ambiguous_query():
    config = ResponsePolicyConfig()
    decision = evaluate_policy(
        user_message="Summarize this tender",
        history=[],
        confidence=0.9,
        estimated_answer_length=200,
        source_filter=None,
        config=config,
    )
    assert decision.action_type == ActionType.ASK_CLARIFY
    assert decision.clarify_questions


def test_summary_then_ask_format_for_long_answer():
    config = ResponsePolicyConfig(length_threshold_summary=300)
    decision = evaluate_policy(
        user_message="Provide a full architecture and rollout plan",
        history=[],
        confidence=0.9,
        estimated_answer_length=1000,
        source_filter=None,
        config=config,
    )
    assert decision.action_type == ActionType.SUMMARY_THEN_ASK_FORMAT
    assert decision.format_prompt


def test_answer_only_when_user_closes():
    config = ResponsePolicyConfig()
    decision = evaluate_policy(
        user_message="Спасибо, все",
        history=[],
        confidence=0.9,
        estimated_answer_length=200,
        source_filter="ted_812-2018_EN.pdf",
        config=config,
    )
    assert decision.action_type == ActionType.ANSWER_ONLY
