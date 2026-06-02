"""
Paired statistical tests on CommitPair collections.

Test: Wilcoxon signed-rank (paired, non-parametric).
Effect size: matched-pairs rank-biserial correlation r.

Why Wilcoxon over paired t-test:
  - commit signals are heavy-tailed and non-normal
  - small sample sizes are likely (< 30 pairs per repo)
  - the signed-rank test makes no distributional assumption
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import wilcoxon

from copilotsig.models import CommitPair, SignalComparison, StudyResult

# Signals to test, in order of theoretical interest
_SIGNALS = [
    # Level A — process structure
    "files_changed",
    "net_lines",
    "total_churn",
    "cross_module_ratio",
    "is_large",
    "is_refactor",
    "touches_tests",
    "test_file_ratio",
    "message_length",
    "has_conventional_commit",
    # Level B — patch content
    "comment_density",
    "docstring_density",
    "type_annotation_ratio",
    "try_density",
    "blank_line_ratio",
]


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation as effect size for Wilcoxon.

    r = 1 - (2 * W_minus) / (n * (n + 1) / 2)
    where W_minus is the sum of ranks of negative differences.
    Equivalent to: r = (W+ - W-) / (W+ + W-)
    """
    diffs = x - y
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = np.argsort(np.abs(diffs)) + 1
    w_pos = float(np.sum(ranks[diffs > 0]))
    w_neg = float(np.sum(ranks[diffs < 0]))
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total > 0 else 0.0


def compare(
    pairs: list[CommitPair],
    scope: str = "all",
) -> StudyResult:
    """
    Run paired Wilcoxon tests on all signals across the given pairs.

    Requires at least 5 pairs per signal to run the test (below this,
    the Wilcoxon test has essentially no power).
    """
    if not pairs:
        return StudyResult(
            scope=scope,
            n_pairs=0,
            n_authors=0,
            interpretation="No pairs available.",
        )

    authors = {p.author for p in pairs}
    comparisons: list[SignalComparison] = []

    for signal in _SIGNALS:
        case_vals = np.array([
            float(getattr(p.case, signal)) for p in pairs
        ])
        ctrl_vals = np.array([
            float(getattr(p.control, signal)) for p in pairs
        ])

        case_med = float(np.median(case_vals))
        ctrl_med = float(np.median(ctrl_vals))
        delta = case_med - ctrl_med
        baseline = max(abs(ctrl_med), 1e-9)

        p_value: Optional[float] = None
        effect: Optional[float] = None

        if len(pairs) >= 5:
            diffs = case_vals - ctrl_vals
            if np.any(diffs != 0):
                try:
                    _, pval = wilcoxon(case_vals, ctrl_vals, alternative="two-sided")
                    p_value = float(pval)
                    effect = _rank_biserial(case_vals, ctrl_vals)
                except ValueError:
                    pass

        if p_value is None or abs(delta) < 1e-9:
            direction = "no_difference"
        elif delta > 0:
            direction = "higher_in_copilot"
        else:
            direction = "lower_in_copilot"

        comparisons.append(SignalComparison(
            signal=signal,
            n_pairs=len(pairs),
            case_median=round(case_med, 4),
            control_median=round(ctrl_med, 4),
            delta_median=round(delta, 4),
            delta_pct=round(delta / baseline * 100, 1),
            p_value=round(p_value, 4) if p_value is not None else None,
            significant_at_05=p_value is not None and p_value < 0.05,
            effect_size=round(effect, 3) if effect is not None else None,
            direction=direction,
        ))

    significant = [c for c in comparisons if c.significant_at_05]
    if not significant:
        interpretation = (
            f"{scope}: No significant differences between Copilot-tagged and "
            f"control commits ({len(pairs)} pairs, {len(authors)} authors). "
            f"H₀ (tagged and untagged commits are drawn from the same distribution) "
            f"cannot be rejected."
        )
    else:
        sig_names = ", ".join(
            f"{c.signal} ({c.direction}, p={c.p_value:.3f})" for c in significant
        )
        interpretation = (
            f"{scope}: {len(significant)}/{len(_SIGNALS)} signals differ significantly "
            f"between Copilot-tagged and control commits "
            f"({len(pairs)} pairs, {len(authors)} authors): {sig_names}. "
            f"Effect sizes are small-to-moderate; confounding by tag-selection bias "
            f"cannot be excluded."
        )

    return StudyResult(
        scope=scope,
        n_pairs=len(pairs),
        n_authors=len(authors),
        signals=comparisons,
        interpretation=interpretation,
    )
