from __future__ import annotations

import logging
import os
from typing import Any

_TOOLS_LOGGER = logging.getLogger("pdf_agent.tools")


def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_tool_start(tool: str, **kwargs: Any) -> None:
    details = " ".join(f"{k}={v}" for k, v in kwargs.items() if v not in (None, ""))
    message = f"tool={tool} status=start"
    if details:
        message = f"{message} {details}"
    _TOOLS_LOGGER.info(message)


def log_tool_success(tool: str, **kwargs: Any) -> None:
    details = " ".join(f"{k}={v}" for k, v in kwargs.items() if v not in (None, ""))
    message = f"tool={tool} status=success"
    if details:
        message = f"{message} {details}"
    _TOOLS_LOGGER.info(message)


def log_tool_error(tool: str, error: Exception) -> None:
    _TOOLS_LOGGER.error("tool=%s status=error error=%s", tool, error, exc_info=True)
