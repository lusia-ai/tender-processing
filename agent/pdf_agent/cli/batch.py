from __future__ import annotations

import argparse
import os
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

from pdf_agent.parsing.chunker import Chunk, iter_chunks

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "tenders" / "raw"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "tenders" / "processed" / "chunks.jsonl"


def main() -> None:
    args = _build_args().parse_args()
    llm_client = None
    llm_model = args.llm_model
    log_fn = print if args.log_llm else None
    if args.strategy == "semantic-llm":
        load_dotenv(dotenv_path=args.env_file, override=False)
        api_key = args.api_key or os.getenv("OPENAI_API_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_TOKEN (or OPENAI_API_KEY) is required for semantic-llm chunking")
        llm_model = args.llm_model or os.getenv("OPENAI_MODEL")
        if not llm_model:
            raise SystemExit("OPENAI_MODEL is required for semantic-llm chunking (or pass --llm-model)")
        llm_client = OpenAI(api_key=api_key)

    input_paths = sorted(Path(args.input_dir).expanduser().resolve().glob("*.pdf"))
    if not input_paths:
        raise SystemExit(f"No PDFs found in {args.input_dir}")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = list(
        iter_chunks(
            input_paths,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            preserve_whitespace=args.preserve_whitespace,
            strategy=args.strategy,
            llm_client=llm_client,
            llm_model=llm_model,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            log_fn=log_fn,
        )
    )
    _write_jsonl(out_path, chunks)

    print(f"Wrote {len(chunks)} chunks from {len(input_paths)} PDFs to {out_path}")
    print(f"Example chunk: {asdict(chunks[0])}")


def _write_jsonl(path: Path, chunks: List[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False))
            f.write("\n")


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch chunk PDFs into JSONL ready for embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Folder with PDFs.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to write JSONL with chunks.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Target chunk size (chars).")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap between chunks (chars).")
    parser.add_argument(
        "--preserve-whitespace",
        action="store_true",
        help="Preserve original whitespace by chunking on characters (no normalization).",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "semantic-llm"],
        default="fixed",
        help="Chunking strategy: fixed-size or semantic (section-aware).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=600,
        help="Minimum token size for semantic-llm chunks.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum token size for semantic-llm chunks.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model for semantic-llm chunking (defaults to OPENAI_MODEL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (fallback to OPENAI_API_TOKEN/OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".env"),
        help="Optional .env file to load for semantic-llm chunking.",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Log LLM boundary checks to stdout.",
    )
    return parser


if __name__ == "__main__":
    main()
