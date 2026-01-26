from __future__ import annotations

import os
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

CHUNK_SIZE = 1024 * 256  # 256 KB


class PDFLoadError(Exception):
    """Raised when a PDF could not be loaded."""


def fetch_pdf(source: str, dest_dir: Optional[Path] = None) -> Tuple[Path, bool]:
    """
    Download a PDF from a URL or validate a local path.

    Returns a tuple of (path, is_temporary).
    """
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        path = _download_pdf(source, dest_dir=dest_dir)
        return path, dest_dir is None

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise PDFLoadError(f"Local path not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Expected a PDF file, got: {path.name}")
    return path, False


def _download_pdf(url: str, dest_dir: Optional[Path]) -> Path:
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PDFLoadError(f"Failed to fetch PDF: {url}") from exc

    suffix = ".pdf"
    parsed = urlparse(url)
    if Path(parsed.path).suffix:
        suffix = Path(parsed.path).suffix

    target_path: Path
    tmp_fd: Optional[int] = None

    if dest_dir is not None:
        dest_dir = dest_dir.expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / _filename_from_url(url, suffix)
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="pdf_", suffix=suffix)
        target_path = Path(tmp_path)

    try:
        if tmp_fd is not None:
            tmp_file_handle = os.fdopen(tmp_fd, "wb")
        else:
            tmp_file_handle = target_path.open("wb")

        with tmp_file_handle as tmp_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    tmp_file.write(chunk)
    except Exception as exc:  # noqa: BLE001
        if dest_dir is None:
            target_path.unlink(missing_ok=True)
        raise PDFLoadError(f"Failed to write PDF for: {url}") from exc

    return target_path


def _filename_from_url(url: str, default_suffix: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        digest = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        name = f"pdf_{digest}{default_suffix}"
    elif Path(name).suffix == "":
        name = f"{name}{default_suffix}"
    return name
