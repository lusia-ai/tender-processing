from __future__ import annotations

from pdf_agent.utils.llm.messages import latest_user_message


def infer_user_language(text: str) -> str:
    return "English"


def state_user_language(state: dict, fallback: str = "") -> str:
    text = latest_user_message(state.get("messages", [])) or fallback
    return infer_user_language(text)


def language_instruction(language: str) -> str:
    return (
        "Respond ONLY in English. Do not answer in other languages even if sources are not in English. "
        "Translate non-English terms to English; you may keep originals in parentheses."
    )


def should_find_similar_query(query: str) -> bool:
    q = query.lower()
    hints = (
        "similar",
        "most similar",
        "closest",
        "nearest",
        "analogous",
        "comparable",
        "matching tenders",
    )
    return any(hint in q for hint in hints)


def should_readiness_query(query: str) -> bool:
    q = query.lower()
    non_readiness_phrases = (
        "qualification requirements",
        "qualification criteria",
        "eligibility criteria",
        "selection criteria",
    )
    if any(phrase in q for phrase in non_readiness_phrases):
        company_cues = ("can we", "are we", "our company", "do we", "we ")
        if not any(cue in q for cue in company_cues):
            return False
    hints = (
        "can we",
        "are we able",
        "are we eligible",
        "are we qualified",
        "do we qualify",
        "can our company",
        "our company",
        "fit for us",
        "bid/no-bid",
        "go/no-go",
        "feasible",
        "readiness",
    )
    return any(hint in q for hint in hints)


def should_chat_history_query(query: str) -> bool:
    q = query.lower()
    hints = (
        "history",
        "chat history",
        "conversation history",
        "summarize chat",
        "summarize conversation",
        "what did we discuss",
        "previous messages",
        "summary of chat",
    )
    return any(hint in q for hint in hints)


def should_force_tender_breakdown(query: str) -> bool:
    q = query.lower()
    hints = (
        "tender breakdown",
        "breakdown",
        "break down",
        "what is in this tender",
        "analyze this tender",
        "tender details",
    )
    section_hints = (
        "buyer",
        "scope",
        "lots",
        "deadlines",
        "qualification requirements",
        "eligibility criteria",
        "securities",
        "payment terms",
        "submission details",
    )
    section_hits = sum(1 for hint in section_hints if hint in q)
    return any(hint in q for hint in hints) or section_hits >= 2
