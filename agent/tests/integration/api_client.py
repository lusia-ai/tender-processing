from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi.testclient import TestClient


def chat_with_file(
    client: TestClient,
    message: str,
    pdf_path: Path,
    top_k: int = 6,
    thread_id: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "message": message,
        "top_k": str(top_k),
        "thread_id": thread_id or str(uuid4()),
    }
    if source_filter:
        payload["source_filter"] = source_filter

    with pdf_path.open("rb") as handle:
        files = {"file": (pdf_path.name, handle, "application/pdf")}
        response = client.post("/chat-file", data=payload, files=files)

    if response.status_code != 200:
        raise AssertionError(f"API call failed ({response.status_code}): {response.text}")
    return response.json()


def find_tool_output(payload: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
    for item in payload.get("tool_outputs", []) or []:
        if isinstance(item, dict) and item.get("kind") == kind:
            return item
    return None
