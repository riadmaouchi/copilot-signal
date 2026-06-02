"""Core pipeline tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from copilotsig.analyzer.pairing import make_pairs
from copilotsig.analyzer.signals import extract
from copilotsig.analyzer.stats import compare
from copilotsig.collector.github import is_copilot_tagged
from copilotsig.models import Commit, CommitFile, Language


COPILOT_MSG = "feat: add config\n\nCo-Authored-By: GitHub Copilot <copilot@github.com>"
PLAIN_MSG   = "fix: handle edge case"
BASE_DATE   = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _commit(sha, author, offset_days, message, n_files=3, net=50):
    return Commit(
        sha=sha, repo="owner/repo", author=author,
        date=BASE_DATE + timedelta(days=offset_days),
        message=message,
        files=[
            CommitFile(filename=f"src/m{i}.py", language=Language.PYTHON,
                       additions=net, deletions=0)
            for i in range(n_files)
        ],
    )


# ── Tag detection ──────────────────────────────────────────────────────────────

def test_copilot_tag_detected():
    assert is_copilot_tagged(COPILOT_MSG)

def test_copilot_tag_case_insensitive():
    assert is_copilot_tagged("co-authored-by: github copilot <copilot@github.com>")

def test_plain_commit_not_tagged():
    assert not is_copilot_tagged(PLAIN_MSG)

def test_model_autodetects_tag():
    c = _commit("x", "alice", 0, COPILOT_MSG)
    assert c.copilot_tagged is True

def test_model_autodetects_no_tag():
    c = _commit("x", "alice", 0, PLAIN_MSG)
    assert c.copilot_tagged is False


# ── Signal extraction ──────────────────────────────────────────────────────────

def test_extract_basic():
    c = _commit("a1", "alice", 0, PLAIN_MSG, n_files=4, net=100)
    s = extract(c)
    assert s.files_changed == 4
    assert s.net_lines == 400
    assert s.is_large is True   # 400 > 200

def test_extract_conventional_commit():
    c = _commit("a2", "alice", 0, "feat: add feature", n_files=1, net=10)
    s = extract(c)
    assert s.has_conventional_commit is True

def test_extract_merge_commit():
    c = _commit("a3", "alice", 0, "Merge branch 'main'", n_files=0, net=0)
    s = extract(c)
    assert s.is_merge is True


# ── Patch content signals ──────────────────────────────────────────────────────

def _commit_with_patch(sha, patch: str, language="python") -> Commit:
    from copilotsig.models import Language as L
    lang = {"python": L.PYTHON, "typescript": L.TYPESCRIPT}.get(language, L.PYTHON)
    return Commit(
        sha=sha, repo="owner/repo", author="alice",
        date=BASE_DATE,
        message=PLAIN_MSG,
        files=[CommitFile(filename="src/foo.py", language=lang,
                          additions=5, deletions=0, patch=patch)],
    )

def test_comment_density_python():
    patch = "\n".join([
        "+def foo():",
        "+    # this is a comment",
        "+    return 1",
        "+# another comment",
    ])
    s = extract(_commit_with_patch("p1", patch))
    assert s.comment_density > 0

def test_type_annotation_ratio_python():
    patch = "\n".join([
        "+def foo(x: int) -> str:",
        "+    return str(x)",
        "+def bar():",
        "+    pass",
    ])
    s = extract(_commit_with_patch("p2", patch))
    assert s.type_annotation_ratio == 0.5   # 1 of 2 functions typed

def test_docstring_density_python():
    patch = "\n".join([
        '+def foo():',
        '+    """Do the thing."""',
        '+    return 1',
        '+def bar():',
        '+    return 2',
    ])
    s = extract(_commit_with_patch("p3", patch))
    assert s.docstring_density == 0.5       # 1 of 2 functions has docstring

def test_try_density_python():
    patch = "\n".join([
        "+try:",
        "+    x = int(s)",
        "+except ValueError:",
        "+    x = 0",
    ])
    s = extract(_commit_with_patch("p4", patch))
    assert s.try_density > 0

def test_blank_line_ratio():
    patch = "\n".join([
        "+def foo():",
        "+    return 1",
        "+",
        "+",
    ])
    s = extract(_commit_with_patch("p5", patch))
    assert s.blank_line_ratio == 0.5        # 2 blank out of 4 added lines

def test_no_patch_signals_are_zero():
    c = _commit("p6", "alice", 0, PLAIN_MSG)  # no patch data
    s = extract(c)
    assert s.comment_density == 0.0
    assert s.docstring_density == 0.0


# ── Pairing ───────────────────────────────────────────────────────────────────

def test_pairing_basic():
    commits = [
        _commit("c1", "alice", 0,  COPILOT_MSG, n_files=5),
        _commit("c2", "alice", 3,  PLAIN_MSG,   n_files=2),
        _commit("c3", "alice", 8,  COPILOT_MSG, n_files=4),
        _commit("c4", "alice", 10, PLAIN_MSG,   n_files=1),
    ]
    signals = [extract(c) for c in commits]
    pairs = make_pairs(signals, max_gap_days=14)
    assert len(pairs) == 2

def test_pairing_no_cross_author():
    commits = [
        _commit("a1", "alice", 0, COPILOT_MSG),
        _commit("b1", "bob",   2, PLAIN_MSG),    # different author — should NOT pair
    ]
    signals = [extract(c) for c in commits]
    pairs = make_pairs(signals, max_gap_days=14)
    assert len(pairs) == 0

def test_pairing_gap_respected():
    commits = [
        _commit("c1", "alice", 0,  COPILOT_MSG),
        _commit("c2", "alice", 20, PLAIN_MSG),   # 20 days > max_gap_days=14
    ]
    signals = [extract(c) for c in commits]
    pairs = make_pairs(signals, max_gap_days=14)
    assert len(pairs) == 0

def test_pairing_no_reuse():
    # Two cases, one control — second case should be unpaired
    commits = [
        _commit("c1", "alice", 0, COPILOT_MSG),
        _commit("c2", "alice", 1, COPILOT_MSG),
        _commit("c3", "alice", 2, PLAIN_MSG),    # only one control
    ]
    signals = [extract(c) for c in commits]
    pairs = make_pairs(signals, max_gap_days=14)
    assert len(pairs) == 1   # control used once only


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_compare_insufficient_pairs():
    result = compare([], scope="test")
    assert result.n_pairs == 0
    assert "No pairs" in result.interpretation

def test_compare_returns_all_signals():
    commits = [
        _commit(f"c{i}", "alice", i * 2,
                COPILOT_MSG if i % 2 == 0 else PLAIN_MSG,
                n_files=3 + (i % 3))
        for i in range(12)
    ]
    signals = [extract(c) for c in commits]
    pairs = make_pairs(signals, max_gap_days=14)
    result = compare(pairs, scope="test")
    assert result.n_pairs == len(pairs)
    signal_names = {s.signal for s in result.signals}
    assert "files_changed" in signal_names
    assert "message_length" in signal_names
