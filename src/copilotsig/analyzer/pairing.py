"""
Case-control pairing: match each Copilot-tagged commit to the nearest
untagged commit from the same author in the same repo.

Design rationale
----------------
We pair within (author, repo) to control for:
- Developer coding style
- Project conventions and language
- Project phase at the time of the commit

The `max_gap_days` parameter limits how far apart pairs can be,
to avoid confounding with seasonal/project-phase variation.
A tight window (≤14 days) is the conservative choice.
"""

from __future__ import annotations

from datetime import timedelta

from copilotsig.models import CommitPair, CommitSignals


def make_pairs(
    signals: list[CommitSignals],
    max_gap_days: int = 14,
) -> list[CommitPair]:
    """
    Build matched pairs from a flat list of CommitSignals.

    Algorithm:
    1. Group by (author, repo).
    2. Within each group, separate Copilot-tagged from control commits.
    3. For each Copilot commit, find the nearest control commit by date
       within max_gap_days. Mark matched controls as used to avoid
       reusing the same control for multiple cases.

    Returns pairs sorted by (repo, author, case.date).
    """
    # Group by (author, repo)
    groups: dict[tuple[str, str], list[CommitSignals]] = {}
    for s in signals:
        key = (s.author, s.repo)
        groups.setdefault(key, []).append(s)

    pairs: list[CommitPair] = []

    for (author, repo), commits in groups.items():
        cases    = sorted([c for c in commits if c.copilot_tagged],  key=lambda x: x.date)
        controls = sorted([c for c in commits if not c.copilot_tagged], key=lambda x: x.date)

        if not cases or not controls:
            continue

        used_control_shas: set[str] = set()

        for case in cases:
            best: CommitSignals | None = None
            best_gap = float("inf")

            for ctrl in controls:
                if ctrl.sha in used_control_shas:
                    continue
                gap = abs((case.date - ctrl.date).total_seconds()) / 86400.0
                if gap <= max_gap_days and gap < best_gap:
                    best = ctrl
                    best_gap = gap

            if best is not None:
                used_control_shas.add(best.sha)
                pairs.append(CommitPair(
                    repo=repo,
                    author=author,
                    case=case,
                    control=best,
                    gap_days=round(best_gap, 2),
                ))

    pairs.sort(key=lambda p: (p.repo, p.author, p.case.date))
    return pairs
