from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver


def _extract_thread_id(config: Any) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        thread_id = configurable.get("thread_id") or configurable.get("thread")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    return None


class BoundedCheckpointer:
    def __init__(self, base: Any, max_threads: int, ttl_seconds: int) -> None:
        self._base = base
        self._max_threads = max_threads
        self._ttl_seconds = ttl_seconds
        self._touch = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _touch_thread(self, config: Any) -> None:
        thread_id = _extract_thread_id(config)
        if not thread_id:
            return
        now = time.time()
        if thread_id in self._touch:
            self._touch.pop(thread_id, None)
        self._touch[thread_id] = now
        self._evict_expired(now)
        self._evict_over_limit()

    def _evict_expired(self, now: float) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff = now - self._ttl_seconds
        for thread_id, ts in list(self._touch.items()):
            if ts >= cutoff:
                continue
            self._touch.pop(thread_id, None)
            self._purge_thread(thread_id)

    def _evict_over_limit(self) -> None:
        if self._max_threads <= 0:
            return
        while len(self._touch) > self._max_threads:
            thread_id, _ = self._touch.popitem(last=False)
            self._purge_thread(thread_id)

    def _purge_thread(self, thread_id: str) -> None:
        for attr in ("storage", "_storage", "checkpoint_storage", "_checkpoint_storage"):
            store = getattr(self._base, attr, None)
            if not isinstance(store, dict):
                continue
            if thread_id in store:
                store.pop(thread_id, None)
                continue
            for key in list(store.keys()):
                if key == thread_id:
                    store.pop(key, None)
                elif isinstance(key, tuple) and thread_id in key:
                    store.pop(key, None)
                elif isinstance(key, str) and thread_id in key:
                    store.pop(key, None)

    def put(self, *args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        self._touch_thread(config)
        return getattr(self._base, "put")(*args, **kwargs)

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        self._touch_thread(config)
        if hasattr(self._base, "aput"):
            return await getattr(self._base, "aput")(*args, **kwargs)
        return getattr(self._base, "put")(*args, **kwargs)

    def get_tuple(self, *args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        self._touch_thread(config)
        return getattr(self._base, "get_tuple")(*args, **kwargs)

    async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        self._touch_thread(config)
        if hasattr(self._base, "aget_tuple"):
            return await getattr(self._base, "aget_tuple")(*args, **kwargs)
        return getattr(self._base, "get_tuple")(*args, **kwargs)


def build_checkpointer(*, max_threads: int, ttl_seconds: int) -> Any:
    base = MemorySaver()
    if max_threads <= 0 and ttl_seconds <= 0:
        return base
    return BoundedCheckpointer(base, max_threads=max_threads, ttl_seconds=ttl_seconds)
