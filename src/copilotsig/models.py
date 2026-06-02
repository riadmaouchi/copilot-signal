"""Core data models for copilot-signal."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Language(str, Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    C          = "c"
    RUBY       = "ruby"
    GO         = "go"
    RUST       = "rust"
    UNKNOWN    = "unknown"


class CommitFile(BaseModel):
    filename: str
    language: Language = Language.UNKNOWN
    additions: int = 0
    deletions: int = 0
    patch: Optional[str] = None
    status: str = "modified"


class Commit(BaseModel):
    sha: str
    repo: str                      # "owner/repo"
    author: str
    date: datetime
    message: str
    files: list[CommitFile] = Field(default_factory=list)
    # True when the commit message contains a Co-Authored-By: Copilot trailer.
    # This is the ground-truth label for the case-control design.
    # Auto-detected from message if not explicitly set.
    copilot_tagged: bool = False

    @model_validator(mode="after")
    def _detect_tag(self) -> "Commit":
        if not self.copilot_tagged:
            from copilotsig.collector.github import is_copilot_tagged
            self.copilot_tagged = is_copilot_tagged(self.message)
        return self

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)


class CommitSignals(BaseModel):
    """
    Per-commit signals — same Level-A set as dev-fingerprint, plus a few
    commit-message signals that become interesting in a paired design.
    """

    sha: str
    repo: str
    author: str
    date: datetime
    copilot_tagged: bool

    # Level A — process
    files_changed: int = 0
    net_lines: int = 0
    total_churn: int = 0
    cross_module_ratio: float = 0.0
    is_refactor: bool = False
    touches_tests: bool = False
    test_file_ratio: float = 0.0
    is_merge: bool = False

    # Commit message
    message_length: int = 0
    has_conventional_commit: bool = False
    # Large commit: |net_lines| > 200 — one boolean, avoids ratio noise at commit level
    is_large: bool = False


class CommitPair(BaseModel):
    """
    A matched pair: one Copilot-tagged commit and one control commit from
    the same author, same repo, within `max_gap_days` calendar days.

    The control is selected as the nearest untagged commit by the same
    author that is not itself within the Copilot window.
    """

    repo: str
    author: str
    case: CommitSignals      # copilot_tagged=True
    control: CommitSignals   # copilot_tagged=False
    gap_days: float          # |case.date - control.date| in days


class SignalComparison(BaseModel):
    """
    Paired statistical comparison for one signal across all CommitPairs.

    Uses Wilcoxon signed-rank test (paired, non-parametric).
    Effect size: matched-pairs rank-biserial correlation.
    """

    signal: str
    n_pairs: int
    case_median: float        # median value in Copilot-tagged commits
    control_median: float     # median value in control commits
    delta_median: float       # case_median - control_median
    delta_pct: float          # delta / max(|control_median|, 1e-9) * 100
    p_value: Optional[float]
    significant_at_05: bool
    effect_size: Optional[float]   # rank-biserial r ∈ [-1, 1]
    direction: str            # "higher_in_copilot", "lower_in_copilot", "no_difference"


class StudyResult(BaseModel):
    """Full results for one repo or the entire corpus."""

    scope: str                # repo name or "all"
    n_pairs: int
    n_authors: int
    signals: list[SignalComparison] = Field(default_factory=list)
    interpretation: str = ""
