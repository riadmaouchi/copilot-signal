"""
Generate synthetic study results for demo and testing.

The synthetic corpus is designed to be honest about what we'd expect:
- commit message length is plausibly higher in Copilot commits
  (Copilot's "generate commit message" feature writes verbose summaries)
- has_conventional_commit is plausibly higher (Copilot follows the format)
- file-level signals (files_changed, net_lines) show NO consistent difference
  because those are driven by the task, not the tool

This is a prior-based simulation, not empirical data.
All synthetic reports are tagged SYNTHETIC in the scope field.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from copilotsig.models import CommitPair, CommitSignals, SignalComparison, StudyResult

random.seed(42)
np.random.seed(42)

SYNTHETIC_REPOS = [
    ("SYNTHETIC/repo-alpha",   80, 0.22),   # (scope, n_pairs, copilot_tag_rate)
    ("SYNTHETIC/repo-beta",    55, 0.18),
    ("SYNTHETIC/repo-gamma",   40, 0.30),
    ("SYNTHETIC/repo-delta",   28, 0.15),
    ("SYNTHETIC/repo-epsilon", 18, 0.25),
]


def _signal_params() -> dict[str, tuple[float, float, float, float, float]]:
    """
    For each signal: (case_mu, ctrl_mu, sigma, true_effect_r, p_approx)

    true_effect_r: rank-biserial r — positive = higher in Copilot commits.
    Signals with |r| < 0.1 are noise (no real effect).
    """
    return {
        # Signals where Copilot plausibly makes a difference
        "message_length":        (72.0, 48.0, 20.0, +0.35, 0.008),   # Copilot writes longer messages
        "has_conventional_commit":(0.82, 0.55, 0.3,  +0.28, 0.022),  # Copilot follows the format

        # Signals where we expect NO consistent difference
        "files_changed":         (3.1,  3.0,  2.5,  +0.04, 0.61),
        "net_lines":             (58.0, 55.0, 80.0, +0.03, 0.72),
        "total_churn":           (95.0, 90.0, 100., +0.03, 0.69),
        "cross_module_ratio":    (0.22, 0.21, 0.18, +0.02, 0.84),
        "is_large":              (0.18, 0.17, 0.3,  +0.04, 0.56),
        "is_refactor":           (0.12, 0.13, 0.2,  -0.03, 0.71),
        "touches_tests":         (0.31, 0.30, 0.35, +0.02, 0.79),
        "test_file_ratio":       (0.14, 0.13, 0.2,  +0.02, 0.81),
    }


def _make_comparison(
    signal: str,
    params: tuple[float, float, float, float, float],
    n_pairs: int,
    noise: float = 1.0,
) -> SignalComparison:
    case_mu, ctrl_mu, sigma, r, p_base = params

    # Add per-repo noise to make repos look different
    case_med = max(0.0, case_mu + np.random.normal(0, sigma * 0.1 * noise))
    ctrl_med = max(0.0, ctrl_mu + np.random.normal(0, sigma * 0.1 * noise))
    delta = case_med - ctrl_med
    baseline = max(abs(ctrl_med), 1e-9)

    # Scale p-value with n_pairs (more pairs → more power)
    power_scale = np.sqrt(n_pairs / 50.0)
    p_scaled = min(1.0, p_base / max(power_scale, 0.1))
    significant = p_scaled < 0.05

    direction = (
        "higher_in_copilot" if delta > 0.01
        else "lower_in_copilot" if delta < -0.01
        else "no_difference"
    )

    return SignalComparison(
        signal=signal,
        n_pairs=n_pairs,
        case_median=round(case_med, 3),
        control_median=round(ctrl_med, 3),
        delta_median=round(delta, 3),
        delta_pct=round(delta / baseline * 100, 1),
        p_value=round(p_scaled, 4),
        significant_at_05=significant,
        effect_size=round(r + np.random.normal(0, 0.05), 3),
        direction=direction,
    )


def generate_results(output_dir: Path) -> list[StudyResult]:
    """Generate synthetic StudyResult objects and write them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    params = _signal_params()
    results: list[StudyResult] = []

    for scope, n_pairs, _ in SYNTHETIC_REPOS:
        noise = random.uniform(0.5, 1.5)
        comparisons = [
            _make_comparison(sig, p, n_pairs, noise)
            for sig, p in params.items()
        ]
        sig_sigs = [c for c in comparisons if c.significant_at_05]
        interp = (
            f"{scope} [SYNTHETIC]: {len(sig_sigs)}/{len(comparisons)} signals significant. "
            + ("message_length and has_conventional_commit differ — consistent with Copilot "
               "generating verbose conventional commit messages. File-level signals: no difference."
               if sig_sigs else "No significant differences detected.")
        )
        result = StudyResult(
            scope=scope,
            n_pairs=n_pairs,
            n_authors=max(3, n_pairs // 15),
            signals=comparisons,
            interpretation=interp,
        )
        results.append(result)
        fname = scope.replace("/", "_") + ".json"
        (output_dir / fname).write_text(result.model_dump_json(indent=2))

    # Also write a pooled result
    total_pairs = sum(n for _, n, _ in SYNTHETIC_REPOS)
    pooled_comparisons = [
        _make_comparison(sig, p, total_pairs, 0.5)
        for sig, p in params.items()
    ]
    sig_sigs = [c for c in pooled_comparisons if c.significant_at_05]
    pooled = StudyResult(
        scope="SYNTHETIC/pooled",
        n_pairs=total_pairs,
        n_authors=sum(max(3, n // 15) for _, n, _ in SYNTHETIC_REPOS),
        signals=pooled_comparisons,
        interpretation=(
            f"SYNTHETIC/pooled: {len(sig_sigs)}/{len(pooled_comparisons)} signals differ "
            f"across {total_pairs} pairs. This is synthetic data for visualization only."
        ),
    )
    results.append(pooled)
    (output_dir / "SYNTHETIC_pooled.json").write_text(pooled.model_dump_json(indent=2))

    # Pair gap distribution (synthetic)
    pairs_meta = []
    for scope, n_pairs, _ in SYNTHETIC_REPOS:
        for _ in range(n_pairs):
            gap = abs(np.random.exponential(4.0))
            gap = min(gap, 14.0)
            pairs_meta.append({"scope": scope, "gap_days": round(gap, 2)})
    (output_dir / "_pairs_meta.json").write_text(json.dumps(pairs_meta, indent=2))

    return results
