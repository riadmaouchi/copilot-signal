"""Extract per-commit signals from enriched Commit objects."""

from __future__ import annotations

import re
from pathlib import Path

from copilotsig.models import Commit, CommitSignals

_MERGE_MSG = re.compile(r"^merge\b", re.IGNORECASE)
_CONVENTIONAL = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?!?:\s",
    re.IGNORECASE,
)
_TEST_PATH = re.compile(
    r"(^|/)tests?/|(^|/)specs?/|(^|/)__tests__/|"
    r"(^|/)test_[^/]+\.(py|go|rs|rb|c|h|js|ts|jsx|tsx)$|"
    r"[^/]+_test\.(py|go|rs|rb|c|h)$|"
    r"[^/]+\.(spec|test)\.(js|ts|jsx|tsx|mjs)$",
    re.IGNORECASE,
)


def _cross_module_ratio(files: list) -> float:
    if len(files) <= 1:
        return 0.0
    top_dirs = {f.filename.split("/")[0] if "/" in f.filename else "root" for f in files}
    return (len(top_dirs) - 1) / (len(files) - 1)


def extract(commit: Commit) -> CommitSignals:
    files = commit.files
    additions = sum(f.additions for f in files)
    deletions = sum(f.deletions for f in files)
    net_lines = additions - deletions
    total_churn = additions + deletions

    test_files = [f for f in files if _TEST_PATH.search(f.filename)]
    msg_first = commit.message.split("\n")[0]

    return CommitSignals(
        sha=commit.sha,
        repo=commit.repo,
        author=commit.author,
        date=commit.date,
        copilot_tagged=commit.copilot_tagged,
        files_changed=len(files),
        net_lines=net_lines,
        total_churn=total_churn,
        cross_module_ratio=_cross_module_ratio(files),
        is_refactor=total_churn > 50 and abs(net_lines) < 0.20 * total_churn,
        touches_tests=len(test_files) > 0,
        test_file_ratio=len(test_files) / max(len(files), 1),
        is_merge=bool(_MERGE_MSG.match(msg_first)),
        is_large=abs(net_lines) > 200,
        message_length=len(msg_first),
        has_conventional_commit=bool(_CONVENTIONAL.match(msg_first)),
    )
