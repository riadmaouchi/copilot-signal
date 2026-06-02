"""GitHub API client — commit search + enrichment."""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from rich.console import Console

from copilotsig.collector.cache import Cache
from copilotsig.models import Commit, CommitFile, Language

_console = Console(stderr=True)

# The exact trailer GitHub Copilot adds when it generates a commit message
# or when its VS Code extension is configured to tag commits.
_COPILOT_PATTERNS = [
    re.compile(r"co-authored-by:.*github copilot", re.IGNORECASE),
    re.compile(r"co-authored-by:.*copilot@github\.com", re.IGNORECASE),
]

_EXT_TO_LANG: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".c": Language.C,
    ".h": Language.C,
    ".rb": Language.RUBY,
    ".go": Language.GO,
    ".rs": Language.RUST,
}


def _detect_language(filename: str) -> Language:
    return _EXT_TO_LANG.get(Path(filename).suffix.lower(), Language.UNKNOWN)


def is_copilot_tagged(message: str) -> bool:
    return any(p.search(message) for p in _COPILOT_PATTERNS)


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        cache: Optional[Cache] = None,
    ) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._cache = cache or Cache()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.AsyncClient(
            base_url=self.BASE,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        self._last_req: float = 0.0

    async def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        key = f"{path}?{params}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gap = time.monotonic() - self._last_req
        if gap < 0.072:
            await asyncio.sleep(0.072 - gap)

        resp = await self._client.get(path, params=params)
        self._last_req = time.monotonic()

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            wait = max(int(resp.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()), 1)
            _console.print(f"[yellow]Rate limited — waiting {wait}s[/yellow]")
            await asyncio.sleep(wait)
            resp = await self._client.get(path, params=params)

        resp.raise_for_status()
        data = resp.json()
        self._cache.set(key, data)
        return data

    # ── Commit search ──────────────────────────────────────────────────────────

    async def search_copilot_repos(
        self,
        min_stars: int = 10,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Find repos that have a significant number of Copilot-tagged commits.

        Strategy: search commits containing the Copilot co-authorship trailer,
        aggregate by repo, return the top repos by hit count.
        """
        repos: dict[str, int] = {}
        page = 1

        while len(repos) < max_results:
            try:
                data = await self._get(
                    "/search/commits",
                    {
                        "q": "co-authored-by:copilot@github.com",
                        "per_page": 100,
                        "page": page,
                        "sort": "committer-date",
                        "order": "desc",
                    },
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 422:
                    break
                raise

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                repo = item.get("repository", {}).get("full_name", "")
                if repo:
                    repos[repo] = repos.get(repo, 0) + 1

            if len(items) < 100:
                break
            page += 1

        return sorted(
            [{"repo": r, "copilot_commits": c} for r, c in repos.items()],
            key=lambda x: x["copilot_commits"],
            reverse=True,
        )

    async def iter_repo_commits(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_commits: int = 2000,
    ) -> AsyncIterator[Commit]:
        """
        Yield all commits in a repo, enriched with file-level data.
        Copilot-tagged commits are flagged via commit.copilot_tagged.
        """
        fetched = 0
        page = 1
        params: dict[str, Any] = {"per_page": 100}
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        while fetched < max_commits:
            params["page"] = page
            batch = await self._get(f"/repos/{owner}/{repo}/commits", params)
            if not batch:
                break

            for raw in batch:
                if fetched >= max_commits:
                    return
                commit = await self._enrich(owner, repo, raw)
                if commit:
                    yield commit
                    fetched += 1

            if len(batch) < 100:
                break
            page += 1

    async def _enrich(
        self, owner: str, repo: str, raw: dict[str, Any]
    ) -> Optional[Commit]:
        sha = raw["sha"]
        detail = await self._get(f"/repos/{owner}/{repo}/commits/{sha}")

        author_info = detail.get("commit", {}).get("author", {})
        date_str = author_info.get("date")
        if not date_str:
            return None

        message = detail.get("commit", {}).get("message", "")

        files: list[CommitFile] = []
        for f in detail.get("files", []):
            files.append(CommitFile(
                filename=f["filename"],
                language=_detect_language(f["filename"]),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch"),
                status=f.get("status", "modified"),
            ))

        return Commit(
            sha=sha,
            repo=f"{owner}/{repo}",
            author=author_info.get("name", "unknown"),
            date=datetime.fromisoformat(date_str.replace("Z", "+00:00")),
            message=message,
            files=files,
            copilot_tagged=is_copilot_tagged(message),
        )

    async def close(self) -> None:
        await self._client.aclose()
        self._cache.close()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
