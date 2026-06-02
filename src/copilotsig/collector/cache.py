"""Disk cache for GitHub API responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import diskcache

_DEFAULT = Path.home() / ".cache" / "copilot-signal"


class Cache:
    def __init__(self, path: Optional[Path] = None, ttl: int = 86400 * 7) -> None:
        d = path or _DEFAULT
        d.mkdir(parents=True, exist_ok=True)
        self._c: diskcache.Cache = diskcache.Cache(str(d))
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        return self._c.get(key)

    def set(self, key: str, value: Any) -> None:
        self._c.set(key, value, expire=self._ttl)

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
