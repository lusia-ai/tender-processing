from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from pdf_agent.parsing.parser import PageText, extract_pages_text
from pdf_agent.prompts.loader import render_prompt

try:
    import tiktoken  # type: ignore
except Exception:  # noqa: BLE001
    tiktoken = None

@dataclass
class Chunk:
    type: str
    source: str  # PDF name
    chunk_id: str
    page: int
    pages: List[int]
    order: int
    text: str
    chars: int


def chunk_pdf(
    path: Path,
    chunk_size: int = 800,
    overlap: int = 120,
    preserve_whitespace: bool = False,
) -> List[Chunk]:
    """Split PDF text into overlapping chunks with metadata."""
    pages = extract_pages_text(path, normalize_whitespace=not preserve_whitespace)
    chunks: List[Chunk] = []

    for page in pages:
        segments = _split_text(page.text, chunk_size, overlap, preserve_whitespace=preserve_whitespace)
        for order, segment in enumerate(segments):
            if not segment:
                continue
            chunk_id = _chunk_id(path, page.page_number, order, segment)
            chunks.append(
                Chunk(
                    type="chunk",
                    source=path.name,
                    chunk_id=chunk_id,
                    page=page.page_number,
                    pages=[page.page_number],
                    order=order,
                    text=segment,
                    chars=len(segment),
                )
            )
    return chunks


def semantic_chunk_pdf(
    path: Path,
    chunk_size: int = 1200,
    overlap: int = 150,
    preserve_whitespace: bool = True,
) -> List[Chunk]:
    """Chunk by section/lot headings, falling back to char splitting for long blocks."""
    pages = extract_pages_text(path, normalize_whitespace=False)
    lines: List[tuple[str, int]] = []
    for page in pages:
        if not page.text:
            continue
        for line in page.text.splitlines():
            lines.append((line, page.page_number))

    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_pages: List[int] = []
    order = 0

    def flush_segment() -> None:
        nonlocal order
        if not current_lines:
            return
        text = "\n".join(current_lines)
        segments = _split_text(text, chunk_size, overlap, preserve_whitespace=preserve_whitespace)
        if not segments:
            return
        pages_sorted = sorted(set(current_pages)) if current_pages else []
        page = pages_sorted[0] if pages_sorted else 1
        for segment in segments:
            if not segment:
                continue
            chunk_id = _chunk_id(path, page, order, segment)
            chunks.append(
                Chunk(
                    type="chunk",
                    source=path.name,
                    chunk_id=chunk_id,
                    page=page,
                    pages=pages_sorted,
                    order=order,
                    text=segment,
                    chars=len(segment),
                )
            )
            order += 1

    for line, page_num in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_major_heading(stripped) and current_lines:
            flush_segment()
            current_lines = [line]
            current_pages = [page_num]
        else:
            current_lines.append(line)
            current_pages.append(page_num)

    flush_segment()
    return chunks


def llm_semantic_chunk_pdf(
    path: Path,
    client,
    model: str,
    min_tokens: int = 600,
    max_tokens: int = 1200,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Chunk]:
    """LLM-driven semantic chunking by topic change detection."""
    blocks = _extract_atomic_blocks(path)
    if not blocks:
        return []

    chunks: List[Chunk] = []
    current_text = ""
    current_pages: List[int] = []
    order = 0

    if log_fn:
        log_fn(f"[llm-chunk] {path.name}: {len(blocks)} atomic blocks")

    for idx, block in enumerate(blocks):
        if not current_text:
            current_text = block.text
            current_pages = block.pages
            continue

        candidate_text = f"{current_text}\n{block.text}"
        if _token_count(candidate_text, model) > max_tokens and _token_count(current_text, model) >= min_tokens:
            chunks.append(_make_chunk(path, current_text, current_pages, order))
            order += 1
            current_text = block.text
            current_pages = block.pages
            continue

        is_new_topic = _llm_is_new_topic(
            client,
            model,
            blocks[idx - 1].text,
            block.text,
            idx=idx,
            log_fn=log_fn,
        )
        if is_new_topic and _token_count(current_text, model) >= min_tokens:
            chunks.append(_make_chunk(path, current_text, current_pages, order))
            order += 1
            current_text = block.text
            current_pages = block.pages
        else:
            current_text = candidate_text
            current_pages = sorted(set(current_pages + block.pages))

    if current_text:
        chunks.append(_make_chunk(path, current_text, current_pages, order))

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int, preserve_whitespace: bool) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if not text:
        return []
    if preserve_whitespace:
        return _split_text_chars(text, chunk_size, overlap)

    words = text.split()
    segments: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        # +1 for space when joining (except first word)
        add_len = len(word) + (1 if current else 0)
        if current_len + add_len > chunk_size and current:
            segments.append(" ".join(current))
            # start new segment with overlap
            if overlap > 0:
                overlap_words = _take_overlap(current, overlap)
                current = overlap_words + [word]
                current_len = sum(len(w) + 1 for w in current) - 1
            else:
                current = [word]
                current_len = len(word)
        else:
            current.append(word)
            current_len += add_len

    if current:
        segments.append(" ".join(current))
    return segments


def _split_text_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size when preserving whitespace")
    segments: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        segments.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return segments


def _take_overlap(words: List[str], overlap: int) -> List[str]:
    """Take trailing words until reaching overlap chars."""
    acc: List[str] = []
    total = 0
    for word in reversed(words):
        add_len = len(word) + (1 if acc else 0)
        if total + add_len > overlap:
            break
        acc.append(word)
        total += add_len
    return list(reversed(acc))


def _chunk_id(path: Path, page: int, order: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{path.stem}_p{page}_c{order}_{digest}"


def iter_chunks(
    paths: Iterable[Path],
    chunk_size: int = 800,
    overlap: int = 120,
    preserve_whitespace: bool = False,
    strategy: str = "fixed",
    llm_client=None,
    llm_model: Optional[str] = None,
    min_tokens: int = 600,
    max_tokens: int = 1200,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Iterable[Chunk]:
    for path in paths:
        if strategy == "semantic":
            yield from semantic_chunk_pdf(
                path,
                chunk_size=chunk_size,
                overlap=overlap,
                preserve_whitespace=preserve_whitespace,
            )
        elif strategy == "semantic-llm":
            if llm_client is None:
                raise ValueError("LLM client required for semantic-llm strategy")
            if not llm_model:
                raise ValueError("LLM model required for semantic-llm strategy")
            yield from llm_semantic_chunk_pdf(
                path,
                client=llm_client,
                model=llm_model,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                log_fn=log_fn,
            )
        else:
            yield from chunk_pdf(
                path,
                chunk_size=chunk_size,
                overlap=overlap,
                preserve_whitespace=preserve_whitespace,
            )


def _is_major_heading(line: str) -> bool:
    patterns = [
        r"^Section\s+[IVX]+:",
        r"^Sekcja\s+[IVX]+:",
        r"^II\.2\.\s*Description",
        r"^II\.2\.\s*Opis",
    ]
    return any(re.match(pat, line, re.IGNORECASE) for pat in patterns)


@dataclass
class AtomicBlock:
    text: str
    pages: List[int]


def _extract_atomic_blocks(path: Path) -> List[AtomicBlock]:
    pages = extract_pages_text(path, normalize_whitespace=False)
    lines: List[Tuple[str, int]] = []
    for page in pages:
        if not page.text:
            continue
        for line in page.text.splitlines():
            lines.append((line, page.page_number))

    blocks: List[AtomicBlock] = []
    pending_heading: List[str] = []
    current_lines: List[str] = []
    current_pages: List[int] = []

    def flush_current() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append(AtomicBlock(text=text, pages=sorted(set(current_pages))))

    for line, page_num in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                flush_current()
                current_lines = []
                current_pages = []
            continue

        if _is_major_heading(stripped):
            if current_lines:
                flush_current()
                current_lines = []
                current_pages = []
            pending_heading.append(stripped)
            continue

        if _is_list_item(stripped):
            if current_lines:
                flush_current()
                current_lines = []
                current_pages = []
            current_lines = pending_heading + [stripped]
            current_pages = [page_num]
            pending_heading = []
            continue

        if not current_lines:
            current_lines = pending_heading + [stripped]
            current_pages = [page_num]
            pending_heading = []
        else:
            current_lines.append(stripped)
            current_pages.append(page_num)

    if current_lines:
        flush_current()
    elif pending_heading:
        blocks.append(AtomicBlock(text="\n".join(pending_heading), pages=[]))

    return blocks


def _is_list_item(line: str) -> bool:
    patterns = [
        r"^[-•]\s+",
        r"^\d+\.\s+",
        r"^\d+\)\s+",
        r"^[a-zA-Z]\)\s+",
    ]
    return any(re.match(pat, line) for pat in patterns)


def _make_chunk(path: Path, text: str, pages: List[int], order: int) -> Chunk:
    page = pages[0] if pages else 1
    chunk_id = _chunk_id(path, page, order, text)
    return Chunk(
        type="chunk",
        source=path.name,
        chunk_id=chunk_id,
        page=page,
        pages=pages or [page],
        order=order,
        text=text,
        chars=len(text),
    )


def _token_count(text: str, model: str) -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _llm_is_new_topic(
    client,
    model: str,
    prev_block: str,
    next_block: str,
    idx: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    system = render_prompt("chunker_topic_system.txt")
    user = render_prompt("chunker_topic_user.txt", prev_block=prev_block, next_block=next_block)
    if log_fn:
        log_fn(f"[llm-chunk] check#{idx} model={model} a_chars={len(prev_block)} b_chars={len(next_block)}")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().upper()
    if log_fn:
        log_fn(f"[llm-chunk] check#{idx} -> {answer}")
    return answer.startswith("Y")
