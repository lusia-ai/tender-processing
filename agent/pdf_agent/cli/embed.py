from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from dotenv import load_dotenv


def main() -> None:
    args = _build_args().parse_args()
    load_dotenv(dotenv_path=args.env_file, override=False)

    provider = args.provider
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        model = args.model or os.getenv("OPENAI_EMBED_MODEL")
        if not model:
            raise SystemExit("OPENAI_EMBED_MODEL is required for provider=openai (or pass --model)")
        dimensions = args.dimensions
        if dimensions is None:
            dim_env = os.getenv("OPENAI_EMBED_DIM")
            if not dim_env:
                raise SystemExit("OPENAI_EMBED_DIM is required for provider=openai (or pass --dimensions)")
            try:
                dimensions = int(dim_env)
            except ValueError as exc:
                raise SystemExit("OPENAI_EMBED_DIM must be an integer") from exc
        embedder = _OpenAIEmbedder(
            model=model,
            api_key=args.api_key or _first_env(["OPENAI_API_TOKEN", "OPENAI_API_KEY"]),
            dimensions=dimensions,
        )
    elif provider == "bge":
        if not args.model:
            raise SystemExit("Embedding model is required for provider=bge (pass --model)")
        embedder = _LocalEmbedder(model=args.model)
    else:
        raise SystemExit(f"Unknown provider: {provider}")

    pinecone_index = None
    if args.pinecone_index:
        from pinecone import Pinecone

        api_key = args.pinecone_api_key or _first_env(["PINECONE_TOKEN", "PINECONE_API_KEY"])
        if not api_key:
            raise SystemExit("Pinecone API key required when using --pinecone-index (set env PINECONE_TOKEN)")
        pinecone_client = Pinecone(api_key=api_key)
        pinecone_index = pinecone_client.Index(args.pinecone_index)

    total = 0
    written = 0

    with Path(args.chunks_file).open() as f_in, out_path.open("w", encoding="utf-8") as f_out:
        buffer: List[dict] = []
        for line in f_in:
            obj = json.loads(line)
            buffer.append(obj)
            if len(buffer) >= args.batch_size:
                written += _handle_batch(buffer, embedder, f_out, pinecone_index, args.namespace)
                total += len(buffer)
                buffer.clear()
        if buffer:
            written += _handle_batch(buffer, embedder, f_out, pinecone_index, args.namespace)
            total += len(buffer)

    print(f"Embedded {written}/{total} chunks -> {out_path}")
    if pinecone_index:
        print(f"Upserted to Pinecone index: {args.pinecone_index}")


def _handle_batch(
    batch: Sequence[dict],
    embedder: "_BaseEmbedder",
    f_out,
    pinecone_index,
    namespace: Optional[str],
) -> int:
    texts = [obj["text"] for obj in batch]
    embeds = embedder.embed(texts)
    assert len(embeds) == len(batch)

    # write JSONL
    for obj, embedding in zip(batch, embeds):
        record = {
            "id": obj["chunk_id"],
            "embedding": embedding,
            "metadata": obj,
        }
        f_out.write(json.dumps(record, ensure_ascii=False))
        f_out.write("\n")

    # optional upsert to Pinecone
    if pinecone_index:
        vectors = []
        for obj, embedding in zip(batch, embeds):
            metadata = _sanitize_metadata(obj)
            vectors.append(
                {
                    "id": obj["chunk_id"],
                    "values": embedding,
                    "metadata": metadata,
                }
            )
        pinecone_index.upsert(vectors=vectors, namespace=namespace or None)

    return len(batch)


class _BaseEmbedder:
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class _OpenAIEmbedder(_BaseEmbedder):
    def __init__(self, model: str, api_key: Optional[str], dimensions: Optional[int]) -> None:
        if not api_key:
            raise SystemExit("OPENAI_API_TOKEN (or OPENAI_API_KEY) is required for provider=openai")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(model=self.model, input=list(texts), **kwargs)
        return [row.embedding for row in resp.data]


class _LocalEmbedder(_BaseEmbedder):
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # normalize_embeddings=True to get cosine-friendly vectors
        vectors = self.model.encode(list(texts), batch_size=16, show_progress_bar=False, normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed chunked PDF text and optionally upsert to Pinecone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".env"),
        help="Optional .env file to load (defaults to .env).",
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "tenders" / "processed" / "chunks.jsonl"),
        help="Path to JSONL produced by pdf_agent.batch",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "tenders" / "processed" / "embeddings.jsonl"),
        help="Where to write embeddings JSONL",
    )
    parser.add_argument("--provider", choices=["openai", "bge"], default="openai", help="Embedding provider")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (OpenAI or sentence-transformers model id). For OpenAI, defaults to OPENAI_EMBED_MODEL.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding calls")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (fallback to OPENAI_API_TOKEN/OPENAI_API_KEY env)")
    parser.add_argument(
        "--dimensions",
        type=int,
        help="Override OpenAI embedding dimensions (defaults to OPENAI_EMBED_DIM).",
    )
    parser.add_argument("--pinecone-index", type=str, help="If set, upsert to this Pinecone index")
    parser.add_argument("--pinecone-api-key", type=str, help="Pinecone API key (fallback to PINECONE_TOKEN/PINECONE_API_KEY env)")
    parser.add_argument("--namespace", type=str, help="Pinecone namespace (optional)")
    return parser


def _first_env(keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return None


def _sanitize_metadata(obj: dict) -> dict:
    clean: dict = {}
    for key, value in obj.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            clean[key] = [str(item) for item in value]
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
            continue
        # fallback to string to avoid Pinecone type errors
        clean[key] = str(value)
    return clean


if __name__ == "__main__":
    main()
