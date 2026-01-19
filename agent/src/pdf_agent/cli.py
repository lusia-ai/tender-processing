from __future__ import annotations

import argparse
from pathlib import Path

from pdf_agent.loader import PDFLoadError, fetch_pdf
from pdf_agent.parser import PDFParseError, ParsedPDF, parse_pdf

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "tenders" / "raw"


def main() -> None:
    args = _build_args().parse_args()
    source = args.pdf or args.url

    save_dir: Path | None = None if args.temp else args.save_dir

    try:
        path, is_temp = fetch_pdf(source, dest_dir=save_dir)
    except PDFLoadError as exc:
        raise SystemExit(f"[load-error] {exc}") from exc

    try:
        parsed = parse_pdf(path, max_pages=args.max_pages, preview_chars=args.preview_chars)
    except PDFParseError as exc:
        if is_temp:
            path.unlink(missing_ok=True)
        raise SystemExit(f"[parse-error] {exc}") from exc

    _print_parsed(parsed, args.max_pages)

    if is_temp:
        path.unlink(missing_ok=True)


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and parse a PDF, printing previews to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pdf", help="Local path to a PDF file.")
    source_group.add_argument("--url", help="HTTP(S) URL to a PDF file.")

    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="How many pages to parse (use 0 or negative to parse all).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Number of characters to include in each page preview.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Where to store downloaded PDFs (URLs). Default: {DEFAULT_SAVE_DIR}",
    )
    parser.add_argument(
        "--temp",
        action="store_true",
        help="Use a temporary file for URL downloads instead of saving.",
    )
    return parser


def _print_parsed(parsed: ParsedPDF, max_pages: int) -> None:
    parsed_pages = len(parsed.previews)
    print(f"PDF: {parsed.path}")
    print(f"Total pages: {parsed.page_count} | Parsed pages: {parsed_pages}")
    print(f"Total characters (parsed pages): {parsed.total_chars}")

    for preview in parsed.previews:
        header = f"\n--- Page {preview.page_number}"
        if max_pages is None or max_pages <= 0:
            header += f" of {parsed.page_count}"
        header += f" | {preview.char_count} chars ---"
        print(header)
        print(preview.preview or "[no text detected]")


if __name__ == "__main__":
    main()
