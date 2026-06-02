"""
Grouped meta-analysis: compare signal detectability by instruction-file type.

Hypothesis: repos with explicit coding-style instructions produce Copilot
commits statistically indistinguishable from manual commits — by design.

Groups (defined by LLM instruction file content, checked June 2026):
  STYLE   — full coding style guide present
              → microsoft/duroxide-pg, SamMRoberts/agentic-engineering, sennap/studio-senn
  PROCESS — instruction file present but governs only process/architecture, not style
              → FrancescoCiulla/NLWeb, odlhassan/JIRA-SCRIPT, jhkidd/world-cup-2026-sweepstake
  NONE    — no LLM instruction file
              → michaeljolley/io, Ronrock/Opaline, battedeefly/animated

Method: per signal, per group:
  - weighted mean rank-biserial r  (weights = n_pairs)
  - Fisher combined p              (combines per-repo p-values)

Note on NLWeb: classified as PROCESS but shows 10/15 significant signals —
not because instructions fail, but because of a task-type confound: Copilot
is used for small single-file fixes (median 1 file) while controls are large
multi-file commits (median 6 files). This inflates size signals in the
opposite direction. Flagged explicitly in the output.

Output: reports/group_style.json, group_process.json, group_none.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2

REPORTS = Path("reports")

# ── Group definitions ──────────────────────────────────────────────────────────
GROUP_STYLE: set[str] = {
    "microsoft/duroxide-pg",
    "SamMRoberts/agentic-engineering",
    "sennap/studio-senn",
}
GROUP_PROCESS: set[str] = {
    "FrancescoCiulla/NLWeb",
    "odlhassan/JIRA-SCRIPT",
    "jhkidd/world-cup-2026-sweepstake",
    "IstiN/dmtools-agents",
    "andy-herman/pensieve",
}
# NONE = everything else

TASK_TYPE_OUTLIER = "FrancescoCiulla/NLWeb"  # anomaly: task-type confound, not instruction effect

MIN_PAIRS = 5  # exclude repos below this from meta-analysis (tests have no power)


def load_reports() -> list[dict]:
    results = []
    for f in sorted(REPORTS.glob("*.json")):
        if f.stem.startswith("_") or f.stem in ("summary",):
            continue
        try:
            d = json.loads(f.read_text())
            if "signals" in d:
                results.append(d)
        except Exception:
            pass
    return results


def classify(scope: str) -> str:
    if scope in GROUP_STYLE:
        return "STYLE"
    if scope in GROUP_PROCESS:
        return "PROCESS"
    return "NONE"


def fisher_combine(p_values: list[float]) -> float | None:
    """Fisher's method: χ² = -2 Σ ln(p_i), df = 2k."""
    valid = [p for p in p_values if p is not None and 0 < p <= 1.0]
    if not valid:
        return None
    chi_sq = -2.0 * sum(math.log(p) for p in valid)
    df = 2 * len(valid)
    return float(1.0 - chi2.cdf(chi_sq, df=df))


def meta_analyze(repos: list[dict], group_name: str) -> dict:
    """
    Pool per-signal statistics across repos in a group.

    For each signal:
      weighted_r = Σ(r_i * n_i) / Σ(n_i)
      combined_p = Fisher's method on per-repo p-values
    """
    eligible = [r for r in repos if r["n_pairs"] >= MIN_PAIRS]
    all_repos = [r for r in repos]

    # Collect all signal names
    signal_names: list[str] = []
    for r in eligible:
        for s in r["signals"]:
            if s["signal"] not in signal_names:
                signal_names.append(s["signal"])

    signals_out = []
    for sig_name in signal_names:
        ns, rs, ps = [], [], []
        for repo in eligible:
            sig_map = {s["signal"]: s for s in repo["signals"]}
            s = sig_map.get(sig_name)
            if s and s["effect_size"] is not None:
                ns.append(repo["n_pairs"])
                rs.append(s["effect_size"])
                if s["p_value"] is not None:
                    ps.append(s["p_value"])

        if not ns:
            signals_out.append({
                "signal": sig_name,
                "n_repos": 0,
                "n_pairs_total": 0,
                "weighted_r": None,
                "combined_p": None,
                "significant_at_05": False,
                "n_sig_repos": 0,
            })
            continue

        total_n = sum(ns)
        w_r = sum(r * n for r, n in zip(rs, ns)) / total_n
        combined_p = fisher_combine(ps) if ps else None
        n_sig_repos = sum(
            1 for repo in eligible
            for s in repo["signals"]
            if s["signal"] == sig_name and s["significant_at_05"]
        )

        signals_out.append({
            "signal": sig_name,
            "n_repos": len(ns),
            "n_pairs_total": total_n,
            "weighted_r": round(w_r, 3),
            "combined_p": round(combined_p, 4) if combined_p is not None else None,
            "significant_at_05": combined_p is not None and combined_p < 0.05,
            "n_sig_repos": n_sig_repos,
        })

    n_sig = sum(1 for s in signals_out if s["significant_at_05"])
    total_pairs = sum(r["n_pairs"] for r in eligible)
    scopes = [r["scope"] for r in all_repos]
    eligible_scopes = [r["scope"] for r in eligible]
    outliers = [s for s in eligible_scopes if s == TASK_TYPE_OUTLIER]

    return {
        "group": group_name,
        "scopes": scopes,
        "eligible_scopes": eligible_scopes,
        "outliers": outliers,
        "n_repos_total": len(all_repos),
        "n_repos_eligible": len(eligible),
        "n_pairs_total": total_pairs,
        "n_signals_significant": n_sig,
        "fraction_significant": round(n_sig / len(signals_out), 3) if signals_out else 0,
        "signals": signals_out,
        "note": (
            f"NLWeb flagged as task-type outlier (10/15 sig from size confound, "
            f"not instruction failure)"
            if TASK_TYPE_OUTLIER in eligible_scopes else ""
        ),
    }


def main() -> None:
    reports = load_reports()

    by_group: dict[str, list[dict]] = {"STYLE": [], "PROCESS": [], "NONE": []}
    for r in reports:
        g = classify(r["scope"])
        by_group[g].append(r)

    print(f"\n{'Group':10s}  {'Repos':5s}  {'≥5 pairs':8s}  {'Total pairs':11s}  {'n_sig/15':8s}")
    print("-" * 60)

    for group_name, repos in by_group.items():
        eligible = [r for r in repos if r["n_pairs"] >= MIN_PAIRS]
        total_pairs = sum(r["n_pairs"] for r in eligible)
        n_sig_total = sum(
            sum(1 for s in r["signals"] if s["significant_at_05"])
            for r in eligible
        )
        max_tests = len(eligible) * 15
        print(f"{group_name:10s}  {len(repos):5d}  {len(eligible):8d}  {total_pairs:11d}  "
              f"{n_sig_total}/{max_tests}")

    print()
    for group_name, repos in by_group.items():
        meta = meta_analyze(repos, group_name)
        out_path = REPORTS / f"group_{group_name.lower()}.json"
        out_path.write_text(json.dumps(meta, indent=2))
        print(f"[{group_name}]  {meta['n_signals_significant']}/15 signals significant "
              f"(Fisher combined p < 0.05)  "
              f"→ {out_path}")
        if meta["note"]:
            print(f"  ⚠  {meta['note']}")
        for s in meta["signals"]:
            if s["significant_at_05"]:
                print(f"     ★ {s['signal']:30s}  r={s['weighted_r']:+.3f}  "
                      f"p={s['combined_p']:.4f}  "
                      f"({s['n_sig_repos']}/{s['n_repos']} repos)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
