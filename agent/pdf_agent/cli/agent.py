from __future__ import annotations

import argparse
from pathlib import Path

from pdf_agent.core.agent import run_agent
from pdf_agent.utils.io.env import load_env


def main() -> None:
    args = _build_args().parse_args()
    load_env(args.env_file)
    try:
        result = run_agent(
            query=args.query,
            pinecone_index=args.pinecone_index,
            namespace=args.namespace,
            top_k=args.top_k,
            source_filter=args.source_filter,
            prefer_lots=not args.no_prefer_lots,
            model=args.model,
            embed_model=args.embed_model,
            dimensions=args.dimensions,
            openai_api_key=args.api_key,
            pinecone_api_key=args.pinecone_api_key,
        )
        _print_result(result)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Console RAG agent over Pinecone using LangGraph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--pinecone-index", required=True, help="Pinecone index name.")
    parser.add_argument("--namespace", help="Pinecone namespace.", default=None)
    parser.add_argument("--top-k", type=int, default=6, help="How many chunks to retrieve.")
    parser.add_argument(
        "--source-filter",
        help="Restrict retrieval to a specific source filename (e.g., ted_812-2018_EN.pdf).",
    )
    parser.add_argument("--no-prefer-lots", action="store_true", help="Disable lot-first retrieval.")
    parser.add_argument("--model", default=None, help="OpenAI chat model (required if OPENAI_MODEL is unset).")
    parser.add_argument(
        "--embed-model",
        default=None,
        help="OpenAI embedding model (required if OPENAI_EMBED_MODEL is unset).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions to match Pinecone index (required if OPENAI_EMBED_DIM is unset).",
    )
    parser.add_argument("--api-key", help="OpenAI API key (fallback to env).")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key (fallback to env).")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".env"),
        help="Optional .env file path.",
    )
    return parser


def _print_result(state: dict) -> None:
    print("=== Answer ===")
    print(state.get("answer", ""))
    print("\n=== Sources ===")
    for src in state.get("sources", []):
        label = f"{src.get('source','?')}#p{src.get('page')}" if src.get("page") else src.get("source", "?")
        print(f"- {label} (score={src.get('score'):.3f})")


if __name__ == "__main__":
    main()
