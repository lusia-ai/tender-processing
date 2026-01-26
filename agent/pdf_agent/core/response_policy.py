from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from pdf_agent.utils.config.settings import rag_token_model

try:
    import tiktoken  # type: ignore
except Exception:  # noqa: BLE001
    tiktoken = None


class ActionType(str, Enum):
    ANSWER_ONLY = "ANSWER_ONLY"
    ANSWER_PLUS_FOLLOWUP = "ANSWER_PLUS_FOLLOWUP"
    ASK_CLARIFY = "ASK_CLARIFY"
    SUMMARY_THEN_ASK_FORMAT = "SUMMARY_THEN_ASK_FORMAT"


@dataclass
class ResponsePolicyConfig:
    enable_proactive_followups: bool = True
    max_followup_items: int = 2
    confidence_threshold_clarify: float = 0.3
    length_threshold_summary: int = 900


@dataclass
class PolicyDecision:
    action_type: ActionType
    followups: List[str]
    clarify_questions: List[str]
    clarify_reason: Optional[str]
    default_assumption: Optional[str]
    format_prompt: Optional[str]


def evaluate_policy(
    user_message: str,
    history: List[Dict[str, str]],
    confidence: float,
    estimated_answer_length: int,
    source_filter: Optional[str],
    config: ResponsePolicyConfig,
    data_scope: str = "tender_notices_only",
) -> PolicyDecision:
    language = infer_language(user_message)
    closing = detect_closing_intent(user_message)
    missing_fields = detect_missing_fields(user_message, history, source_filter, data_scope)
    ambiguity_flags = []
    if missing_fields:
        ambiguity_flags.append("missing_fields")
    if confidence < config.confidence_threshold_clarify:
        ambiguity_flags.append("low_confidence")

    if closing:
        return PolicyDecision(
            action_type=ActionType.ANSWER_ONLY,
            followups=[],
            clarify_questions=[],
            clarify_reason=None,
            default_assumption=None,
            format_prompt=None,
        )

    if ambiguity_flags:
        questions, reason, default_assumption = clarify_questions(missing_fields, language, confidence)
        return PolicyDecision(
            action_type=ActionType.ASK_CLARIFY,
            followups=[],
            clarify_questions=questions,
            clarify_reason=reason,
            default_assumption=default_assumption,
            format_prompt=None,
        )

    if estimated_answer_length >= config.length_threshold_summary:
        return PolicyDecision(
            action_type=ActionType.SUMMARY_THEN_ASK_FORMAT,
            followups=[],
            clarify_questions=[],
            clarify_reason=None,
            default_assumption=None,
            format_prompt=format_prompt(language),
        )

    if config.enable_proactive_followups:
        followups = followup_suggestions(user_message, language, config.max_followup_items)
        return PolicyDecision(
            action_type=ActionType.ANSWER_PLUS_FOLLOWUP,
            followups=followups,
            clarify_questions=[],
            clarify_reason=None,
            default_assumption=None,
            format_prompt=None,
        )

    return PolicyDecision(
        action_type=ActionType.ANSWER_ONLY,
        followups=[],
        clarify_questions=[],
        clarify_reason=None,
        default_assumption=None,
        format_prompt=None,
    )


def detect_missing_fields(
    user_message: str,
    history: List[Dict[str, str]],
    source_filter: Optional[str],
    data_scope: str,
) -> List[str]:
    missing: List[str] = []
    tender_ref = extract_tender_reference(user_message) or extract_tender_reference(_history_text(history))
    if _mentions_tender(user_message) and not tender_ref and not source_filter:
        missing.append("tender_id_or_file")

    if _mentions_award_or_delivery(user_message) and data_scope == "tender_notices_only":
        missing.append("data_scope_award")

    if _mentions_comparison(user_message) and not tender_ref:
        missing.append("comparison_target")

    return missing


def clarify_questions(
    missing_fields: List[str],
    language: str,
    confidence: float,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    questions: List[str] = []
    reason = None
    default_assumption = None

    if "tender_id_or_file" in missing_fields:
        questions.append("Which tender or file should I use for this answer?")
        reason = "I want to avoid pulling information from the wrong tender."
        default_assumption = "I can search across all tenders if that works."

    if "data_scope_award" in missing_fields:
        questions.append("Do you need actual award/delivery data or only tender requirements?")
        reason = "The current index contains tender notices only."
        default_assumption = "I can answer based on tender scope if that works."

    if "comparison_target" in missing_fields:
        questions.append("What should I compare it against - another tender or your plant profile?")

    if not questions:
        questions.append("Could you clarify the request so I can be precise?")
        reason = "There isn't enough context for a confident answer."
        default_assumption = "I can provide a broad overview across the index."

    # Limit to 2 questions
    return questions[:2], reason, default_assumption


def followup_suggestions(user_message: str, language: str, max_items: int) -> List[str]:
    suggestions: List[str] = []
    intent = detect_intent(user_message)

    if intent == "timeline":
        suggestions.append("Want me to extract deadlines and milestones?")
    if intent == "requirements":
        suggestions.append("Should I list qualification and compliance requirements?")
    if intent == "pricing":
        suggestions.append("Need an estimated value summary per lot?")
    if intent == "comparison":
        suggestions.append("Should I compare against similar tenders or past awards?")

    if not suggestions:
        suggestions = [
            "Do you want a one-page brief for the bid team?",
            "Should I pull similar tenders for context?",
        ]

    return suggestions[:max_items]


def format_prompt(language: str) -> str:
    return (
        "How should I deliver the full details: one message, step-by-step, "
        "or plan first then implementation? With examples or without?"
    )


def detect_closing_intent(text: str) -> bool:
    patterns = [
        r"\bthanks\b",
        r"\bthat'?s all\b",
        r"\bgot it\b",
    ]
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)


def infer_language(text: str) -> str:
    return "English"


def detect_intent(text: str) -> str:
    if re.search(r"(deadline|timeline|duration|term)", text, re.IGNORECASE):
        return "timeline"
    if re.search(r"(requirement|qualification|criteria|requirements)", text, re.IGNORECASE):
        return "requirements"
    if re.search(r"(price|cost|value|budget)", text, re.IGNORECASE):
        return "pricing"
    if re.search(r"(compare|similar)", text, re.IGNORECASE):
        return "comparison"
    return "general"


def extract_tender_reference(text: str) -> Optional[str]:
    match = re.search(r"(ted_\d+-\d{4}|\b\d{4}-\d{4}\b|\.pdf\b)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _mentions_tender(text: str) -> bool:
    return bool(re.search(r"(tender|lot|zadanie|cz[eę]ść)", text, re.IGNORECASE))


def _mentions_award_or_delivery(text: str) -> bool:
    return bool(re.search(r"(delivered|award|awarded|won)", text, re.IGNORECASE))


def _mentions_comparison(text: str) -> bool:
    return bool(re.search(r"(compare|vs|versus)", text, re.IGNORECASE))


def _history_text(history: List[Dict[str, str]]) -> str:
    return "\n".join(item.get("content", "") for item in history)


def estimate_length(text: str, model: Optional[str] = None) -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)
    if not model:
        model = rag_token_model()
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _t(language: str, english: str) -> str:
    return english
