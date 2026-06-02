"""Extract per-commit signals from enriched Commit objects."""

from __future__ import annotations

import re
from pathlib import Path

from copilotsig.models import Commit, CommitFile, CommitSignals, Language

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


_SUPPORTED = {Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT, Language.GO, Language.RUST}

_PY_DEF    = re.compile(r"^\s*def\s+\w+")
_PY_TYPED  = re.compile(r"^\s*def\s+.*->")
_PY_DOC    = re.compile(r'^\s*("""|\'\'\')' )
_PY_TRY    = re.compile(r"^\s*try\s*:")
_PY_CMT    = re.compile(r"^\s*#")

_JS_FN     = re.compile(r"(function\s+\w+|\b\w+\s*[:=]\s*(async\s+)?(\(.*?\)|[a-zA-Z_]\w*)\s*=>|^\s*(async\s+)?function\b)")
_JS_JSDOC  = re.compile(r"^\s*\*")
_JS_TRY    = re.compile(r"^\s*try\s*\{")
_JS_CMT    = re.compile(r"^\s*//")

_GO_FN     = re.compile(r"^\s*func\s+")
_GO_TRY    = re.compile(r"if\s+err\s*!=\s*nil")
_GO_CMT    = re.compile(r"^\s*//")

_RUST_FN   = re.compile(r"^\s*(pub\s+)?fn\s+\w+")
_RUST_TRY  = re.compile(r"\?\s*;|\.unwrap\(\)|\.expect\(")
_RUST_CMT  = re.compile(r"^\s*//")


def _patch_signals(files: list[CommitFile]) -> dict:
    """Parse added lines in patch diffs and return content signal dict."""
    total_added = 0
    blank_added = 0
    comment_lines = 0
    try_blocks = 0
    fn_defs = 0
    fn_typed = 0
    fn_with_doc = 0

    for f in files:
        if not f.patch or f.language not in _SUPPORTED:
            continue
        lang = f.language

        added: list[str] = []
        for line in f.patch.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:])

        if not added:
            continue

        total_added += len(added)
        in_jsdoc = False

        for i, raw in enumerate(added):
            stripped = raw.strip()
            if not stripped:
                blank_added += 1
                continue

            if lang == Language.PYTHON:
                if _PY_CMT.match(raw):
                    comment_lines += 1
                if _PY_TRY.match(raw):
                    try_blocks += 1
                if _PY_DEF.match(raw):
                    fn_defs += 1
                    if _PY_TYPED.match(raw):
                        fn_typed += 1
                    # docstring on next non-blank added line
                    for j in range(i + 1, min(i + 4, len(added))):
                        nxt = added[j].strip()
                        if nxt and _PY_DOC.match(added[j]):
                            fn_with_doc += 1
                            break
                        elif nxt:
                            break

            elif lang in (Language.JAVASCRIPT, Language.TYPESCRIPT):
                if stripped.startswith('/**'):
                    in_jsdoc = True
                if in_jsdoc:
                    comment_lines += 1
                    if '*/' in stripped:
                        in_jsdoc = False
                elif _JS_CMT.match(raw):
                    comment_lines += 1
                if _JS_TRY.match(raw):
                    try_blocks += 1
                if _JS_FN.search(raw):
                    fn_defs += 1
                    # check for preceding /** in added lines
                    for j in range(i - 1, max(i - 5, -1), -1):
                        ps = added[j].strip()
                        if '/**' in ps:
                            fn_with_doc += 1
                            break
                        elif ps and not _JS_JSDOC.match(added[j]) and '*/' not in ps:
                            break

            elif lang == Language.GO:
                if _GO_CMT.match(raw):
                    comment_lines += 1
                if _GO_TRY.search(raw):
                    try_blocks += 1
                if _GO_FN.match(raw):
                    fn_defs += 1

            elif lang == Language.RUST:
                if _RUST_CMT.match(raw):
                    comment_lines += 1
                if _RUST_TRY.search(raw):
                    try_blocks += 1
                if _RUST_FN.match(raw):
                    fn_defs += 1

    code_lines = total_added - blank_added
    return dict(
        comment_density=comment_lines / max(code_lines, 1),
        docstring_density=fn_with_doc / max(fn_defs, 1),
        type_annotation_ratio=fn_typed / max(fn_defs, 1),
        try_density=try_blocks / max(total_added, 1) * 100,
        blank_line_ratio=blank_added / max(total_added, 1),
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
        **_patch_signals(files),
    )
