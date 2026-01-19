# ted_pull_20.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
import requests


SEARCH_URL = "https://api.ted.europa.eu/v3/notices/search"

def post_with_retry(session: requests.Session, url: str, payload: dict, retries: int = 6) -> dict:
    for attempt in range(retries):
        r = session.post(url, json=payload, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 20))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}\n{r.text}")
        return r.json()
    raise RuntimeError("Too many retries")

def iter_publication_numbers(resp: dict) -> list[str]:
    """
    API response schema can evolve; try a few common shapes.
    """
    candidates = []
    for key in ("results", "notices", "items"):
        if isinstance(resp.get(key), list):
            candidates = resp[key]
            break

    pub_numbers: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        # common keys
        for k in ("publication-number", "publicationNumber", "ND"):
            v = item.get(k)
            if isinstance(v, str) and "-" in v:
                pub_numbers.append(v)
                break
    return pub_numbers

def download_pdf(session: requests.Session, pub_no: str, out_dir: Path, lang: str = "en") -> bool:
    url = f"https://ted.europa.eu/{lang}/notice/{pub_no}/pdf"
    out_path = out_dir / f"{pub_no}_{lang}.pdf"
    r = session.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        return False
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return True

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="TED expert query")
    ap.add_argument("--scope", default="ALL", choices=["LATEST", "ACTIVE", "ALL"])
    ap.add_argument("--need", type=int, default=20)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--out", default="ted_pdfs")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "ted-puller/1.0",
    })

    got: list[str] = []
    page = 1

    while len(got) < args.need:
        payload = {
            "query": args.query,
            "page": page,
            "limit": args.page_size,
            "scope": args.scope,
            "paginationMode": "PAGE_NUMBER",
            "fields": ["publication-number"],  # достаточно для скачивания PDF
        }
        resp = post_with_retry(session, SEARCH_URL, payload)
        pubs = iter_publication_numbers(resp)

        if not pubs:
            break

        for p in pubs:
            if p in got:
                continue
            if download_pdf(session, p, out_dir, lang=args.lang):
                got.append(p)
                print(f"[{len(got)}/{args.need}] downloaded {p}")
                if len(got) >= args.need:
                    break
            else:
                # если у notice нет EN PDF — просто пропускаем
                print(f"skip {p} (no {args.lang} pdf?)")

        page += 1

    print("\nDone. Saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
