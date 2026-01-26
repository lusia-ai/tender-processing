from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class IntegrationConfig:
    openai_api_key: str
    openai_model: str
    openai_embed_model: str
    openai_embed_dim: int
    judge_model: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: Optional[str]
    tender_pdf_path: Path


def load_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def require_env(*keys: str) -> str:
    value = _env(*keys)
    if not value:
        joined = "/".join(keys)
        raise RuntimeError(f"Missing required environment variable: {joined}")
    return value


def build_config() -> IntegrationConfig:
    load_env()
    openai_api_key = require_env("OPENAI_API_TOKEN", "OPENAI_API_KEY")
    openai_model = require_env("OPENAI_MODEL")
    openai_embed_model = require_env("OPENAI_EMBED_MODEL")
    openai_embed_dim = int(require_env("OPENAI_EMBED_DIM"))
    judge_model = _env("OPENAI_JUDGE_MODEL") or openai_model
    pinecone_api_key = require_env("PINECONE_TOKEN", "PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX", "PINECONE_INDEX_NAME")
    pinecone_namespace = _env("PINECONE_NAMESPACE")
    tender_pdf_path = Path(__file__).resolve().parents[2] / "data" / "tenders" / "raw" / "ted_5791-2018_EN.pdf"
    if not tender_pdf_path.exists():
        raise RuntimeError(f"Tender PDF not found: {tender_pdf_path}")
    return IntegrationConfig(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_embed_model=openai_embed_model,
        openai_embed_dim=openai_embed_dim,
        judge_model=judge_model,
        pinecone_api_key=pinecone_api_key,
        pinecone_index=pinecone_index,
        pinecone_namespace=pinecone_namespace,
        tender_pdf_path=tender_pdf_path,
    )
