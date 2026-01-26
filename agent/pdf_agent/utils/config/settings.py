from __future__ import annotations

import os

from pdf_agent.utils.io.env import require_env


def min_score() -> float:
    return float(os.getenv("MIN_SCORE", "0.2"))


def max_per_source() -> int:
    return int(os.getenv("MAX_PER_SOURCE", "3"))


def max_rag_tokens() -> int:
    return int(os.getenv("MAX_RAG_TOKENS", "1600"))


def max_upload_tokens() -> int:
    return int(os.getenv("MAX_UPLOAD_TOKENS", "1200"))


def upload_chunk_size() -> int:
    return int(os.getenv("UPLOAD_CHUNK_SIZE", "800"))


def upload_chunk_overlap() -> int:
    return int(os.getenv("UPLOAD_CHUNK_OVERLAP", "120"))


def rag_token_model() -> str:
    return require_env("OPENAI_MODEL")


def enable_similar_retrieval() -> bool:
    return os.getenv("ENABLE_SIMILAR_RETRIEVAL", "true").lower() in {"1", "true", "yes"}


def enable_doc_query_building() -> bool:
    return os.getenv("ENABLE_DOC_QUERY_BUILDING", "true").lower() in {"1", "true", "yes"}


def doc_query_max() -> int:
    return int(os.getenv("DOC_QUERY_MAX", "4"))


def doc_query_tokens() -> int:
    return int(os.getenv("DOC_QUERY_TOKENS", "800"))


def similar_top_k(base_top_k: int) -> int:
    default_k = max(6, base_top_k * 3)
    return int(os.getenv("SIMILAR_TOP_K", str(default_k)))


def similar_max_per_source() -> int:
    return int(os.getenv("SIMILAR_MAX_PER_SOURCE", "1"))


def similar_total() -> int:
    return int(os.getenv("SIMILAR_TOTAL", "12"))


def enable_query_rewrite() -> bool:
    return os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in {"1", "true", "yes"}


def rewrite_max_messages() -> int:
    return int(os.getenv("REWRITE_MAX_MESSAGES", "6"))


def max_history_tokens() -> int:
    return int(os.getenv("MAX_HISTORY_TOKENS", "1200"))


def default_model() -> str:
    return require_env("OPENAI_MODEL")


def default_embed_model() -> str:
    return require_env("OPENAI_EMBED_MODEL")


def default_embed_dim() -> int:
    return int(require_env("OPENAI_EMBED_DIM"))


def enable_auto_similar() -> bool:
    return os.getenv("ENABLE_AUTO_SIMILAR", "false").lower() in {"1", "true", "yes"}
