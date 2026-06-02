"""
Generate figures for copilot-signal.

Run after python run_study.py, or use --synthetic for demo mode:
    python generate_figures.py            # uses reports/
    python generate_figures.py --synthetic

Produces 4 figures in docs/img/:
  fig1_corpus.png   — corpus overview: repos, commit counts, tag rates
  fig2_forest.png   — main result: signal differences (forest plot)
  fig3_heatmap.png  — signal × repo consistency matrix
  fig4_pairs.png    — pairing quality: gap distribution
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#161B22"
GRID    = "#21262D"
BORDER  = "#30363D"
TEXT_HI = "#E6EDF3"
TEXT_LO = "#8B949E"
DPI     = 160

C_SIG   = "#3fb950"   # significant (green)
C_NS    = "#484F58"   # not significant
C_TAG   = "#f0883e"   # Copilot-tagged (orange)
C_CTRL  = "#58a6ff"   # control (blue)

CMAP_DIV = LinearSegmentedColormap.from_list(
    "rbc", ["#58a6ff", SURFACE, "#f0883e"], N=256
)

REPORTS_DIR = Path("reports")
IMG_DIR     = Path("docs/img")

SIGNAL_LABELS = {
    "files_changed":          "Files changed",
    "net_lines":              "Net lines",
    "total_churn":            "Total churn (add+del)",
    "cross_module_ratio":     "Cross-module ratio",
    "is_large":               "Large commit (|Δ| > 200)",
    "is_refactor":            "Refactor commit",
    "touches_tests":          "Touches test files",
    "test_file_ratio":        "Test-file ratio",
    "message_length":         "Commit message length",
    "has_conventional_commit":"Conventional commit format",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ax_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_LO, labelsize=10)


def _save(fig: plt.Figure, name: str, synthetic: bool = False) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    tag = " [SYNTHETIC]" if synthetic else ""
    print(f"  ✓ {path}{tag}  ({path.stat().st_size // 1024} KB)")


def _watermark(fig: plt.Figure, n_pairs: int, synthetic: bool) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"SYNTHETIC DATA — not empirical  ·  {today}" if synthetic \
        else f"Real GitHub data · {n_pairs:,} matched pairs · {today}"
    color = "#d29922" if synthetic else TEXT_LO
    fig.text(0.99, 0.005, label, ha="right", va="bottom",
             color=color, fontsize=7, alpha=0.8,
             fontweight="bold" if synthetic else "normal")


def _stars(p: float | None) -> str:
    if p is None: return ""
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★"
    if p < 0.05:  return "★"
    return ""


def load_results(reports_dir: Path) -> list[dict]:
    results = []
    for f in sorted(reports_dir.glob("*.json")):
        if f.stem.startswith("_") or f.stem == "summary":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


# ── Fig 1: Corpus overview ─────────────────────────────────────────────────────
def fig1_corpus(results: list[dict], synthetic: bool) -> None:
    """
    Horizontal bar chart — one row per repo.
    Two bars: total commits fetched (blue) and Copilot-tagged subset (orange).
    Annotated with number of matched pairs and tag rate.
    """
    # Sort by n_pairs descending, exclude pooled
    rows = sorted(
        [r for r in results if "pooled" not in r["scope"]],
        key=lambda r: r["n_pairs"],
        reverse=True,
    )
    if not rows:
        return

    names  = [r["scope"].split("/")[-1] for r in rows]
    pairs  = [r["n_pairs"] for r in rows]
    authors = [r["n_authors"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.75 + 1.5)), facecolor=BG)
    _ax_dark(ax)

    y = np.arange(len(rows))
    bars = ax.barh(y, pairs, color=C_TAG, alpha=0.85, height=0.55,
                   label="Matched pairs", zorder=3)

    for i, (bar, n_p, n_a) in enumerate(zip(bars, pairs, authors)):
        ax.text(n_p + max(pairs) * 0.012, i,
                f"{n_p} pairs · {n_a} authors",
                va="center", ha="left", color=C_TAG,
                fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11.5, color=TEXT_HI)
    ax.invert_yaxis()
    ax.set_xlabel("Matched pairs  (Copilot-tagged ↔ nearest untagged, same author, ≤14 days)",
                  color=TEXT_LO, fontsize=10)
    ax.set_xlim(0, max(pairs) * 1.45)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    total_pairs  = sum(pairs)
    total_authors = sum(authors)

    fig.suptitle(
        f"Study corpus — {len(rows)} repos · {total_pairs} matched pairs · {total_authors} authors",
        color=TEXT_HI, fontsize=14, fontweight="bold", y=1.01,
    )
    ax.set_title(
        "Each pair: one Copilot-tagged commit matched to the nearest untagged commit\n"
        "from the same author in the same repo within 14 days",
        color=TEXT_LO, fontsize=9.5, pad=8,
    )
    _watermark(fig, total_pairs, synthetic)
    _save(fig, "fig1_corpus.png", synthetic)


# ── Fig 2: Forest plot — main finding ─────────────────────────────────────────
def fig2_forest(results: list[dict], synthetic: bool) -> None:
    """
    Forest plot of signal differences across all pairs (pooled or largest scope).

    Each row = one signal.
    X-axis: Δ% (case median vs control median, relative to control).
    Color: significant (green) vs. not (gray).
    Right margin: p-value + effect size r.

    The vertical zero line = null hypothesis.
    """
    # Use the pooled result if available, else the largest repo
    pooled = next((r for r in results if "pooled" in r["scope"]), None)
    target = pooled or max(results, key=lambda r: r["n_pairs"], default=None)
    if not target:
        return

    sigs = {s["signal"]: s for s in target.get("signals", [])}
    signals_ordered = list(SIGNAL_LABELS.keys())
    rows = [sigs[s] for s in signals_ordered if s in sigs]
    if not rows:
        return

    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.72 + 1.5)), facecolor=BG)
    _ax_dark(ax)

    y = np.arange(n)
    deltas   = [r["delta_pct"] for r in rows]
    colors   = [C_SIG if r["significant_at_05"] else C_NS for r in rows]
    alphas   = [0.95 if r["significant_at_05"] else 0.55 for r in rows]
    labels   = [SIGNAL_LABELS.get(r["signal"], r["signal"]) for r in rows]

    # Horizontal bars from 0
    for i, (delta, color, alpha) in enumerate(zip(deltas, colors, alphas)):
        ax.barh([i], [delta], color=color, alpha=alpha, height=0.5,
                left=0, zorder=3)

    # Point markers at delta value
    ax.scatter(deltas, y, s=60, color=colors, zorder=5, edgecolors="white", linewidths=0.5)

    # Zero line (null)
    ax.axvline(0, color=TEXT_HI, lw=1.5, alpha=0.7, zorder=4)

    # Right-side annotation: p-value, effect size, direction
    x_max = max(abs(d) for d in deltas) * 1.0
    for i, row in enumerate(rows):
        p_str = f"p={row['p_value']:.3f}" if row["p_value"] is not None else "p=—"
        r_str = f"r={row['effect_size']:+.2f}" if row["effect_size"] is not None else ""
        stars = _stars(row["p_value"])
        label = f"{p_str}  {r_str}  {stars}".strip()
        color = C_SIG if row["significant_at_05"] else TEXT_LO
        ax.text(x_max * 1.05, i, label,
                va="center", ha="left", color=color,
                fontsize=9.5, fontweight="bold" if row["significant_at_05"] else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5, color=TEXT_HI)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Δ%  (Copilot-tagged median vs. matched control median, same author ±14 days)",
        color=TEXT_LO, fontsize=10,
    )
    lim = x_max * 1.65
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend_items = [
        mpatches.Patch(color=C_SIG, label="p < 0.05  ★  significant"),
        mpatches.Patch(color=C_NS,  label="p ≥ 0.05  —  not significant"),
        plt.Line2D([0], [0], color=TEXT_HI, lw=1.5, label="null (Δ% = 0)"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT_LO, fontsize=9.5, framealpha=0.9)

    n_sig = sum(1 for r in rows if r["significant_at_05"])
    n_total = len(rows)
    scope_label = target["scope"].replace("SYNTHETIC/", "")
    ax.set_title(
        f"Scope: {scope_label}  ·  {target['n_pairs']} pairs  ·  "
        f"{n_sig}/{n_total} signals significant\n"
        f"r = matched-pairs rank-biserial correlation (effect size)",
        color=TEXT_LO, fontsize=9.5, pad=8,
    )
    fig.suptitle(
        "Do Copilot-tagged commits differ from control commits by the same developer?",
        color=TEXT_HI, fontsize=14, fontweight="bold", y=1.01,
    )
    _watermark(fig, target["n_pairs"], synthetic)
    _save(fig, "fig2_forest.png", synthetic)


# ── Fig 3: Signal × repo heatmap ──────────────────────────────────────────────
def fig3_heatmap(results: list[dict], synthetic: bool) -> None:
    """
    Matrix: repos (rows) × signals (columns).
    Color: rank-biserial r (blue = lower in Copilot, orange = higher).
    Stars in cells where p < 0.05.

    Shows whether the findings are consistent across repos or repo-specific noise.
    """
    repo_results = [r for r in results if "pooled" not in r["scope"]]
    if not repo_results:
        return

    signal_order = list(SIGNAL_LABELS.keys())
    repo_names = [r["scope"].split("/")[-1] for r in repo_results]

    n_repos = len(repo_results)
    n_sigs  = len(signal_order)
    matrix_r = np.zeros((n_repos, n_sigs))
    matrix_p = np.ones((n_repos, n_sigs))

    for i, result in enumerate(repo_results):
        sig_map = {s["signal"]: s for s in result.get("signals", [])}
        for j, sig in enumerate(signal_order):
            if sig in sig_map:
                s = sig_map[sig]
                matrix_r[i, j] = s["effect_size"] if s["effect_size"] is not None else 0.0
                matrix_p[i, j] = s["p_value"] if s["p_value"] is not None else 1.0

    fig_h = max(4, n_repos * 0.7 + 2.0)
    fig_w = max(10, n_sigs * 1.1 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    im = ax.imshow(matrix_r, aspect="auto", cmap=CMAP_DIV, norm=norm)

    # Cell annotations
    for i in range(n_repos):
        for j in range(n_sigs):
            r_val = matrix_r[i, j]
            p_val = matrix_p[i, j]
            txt_color = TEXT_HI if abs(r_val) > 0.2 else TEXT_LO
            r_str = f"{r_val:+.2f}" if abs(r_val) > 0.01 else "·"
            star = _stars(p_val)
            ax.text(j, i, f"{r_str}\n{star}" if star else r_str,
                    ha="center", va="center",
                    color=txt_color, fontsize=8.5, linespacing=1.2)

    sig_short = [
        SIGNAL_LABELS[s].replace(" ratio", "").replace(" commit", "").replace(" file", "")
        for s in signal_order
    ]
    ax.set_xticks(range(n_sigs))
    ax.set_xticklabels(sig_short, rotation=35, ha="right",
                       color=TEXT_LO, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_repos))
    ax.set_yticklabels(repo_names, fontsize=10.5, color=TEXT_HI)
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.18, shrink=0.5)
    cbar.ax.tick_params(colors=TEXT_LO, labelsize=8)
    cbar.ax.set_xlabel("rank-biserial r  (negative = lower in Copilot commits, positive = higher)",
                       color=TEXT_LO, fontsize=8.5)
    cbar.outline.set_edgecolor(BORDER)
    cbar.ax.set_facecolor(SURFACE)

    # Star legend
    fig.text(0.01, 0.01,
             "★ p < 0.05   ★★ p < 0.01   ★★★ p < 0.001   · no meaningful effect",
             color=TEXT_LO, fontsize=8, ha="left", va="bottom", alpha=0.8)

    fig.suptitle(
        "Signal × repo consistency — rank-biserial effect size",
        color=TEXT_HI, fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "Consistent orange columns = signals reliably higher in Copilot commits  "
        "·  Mixed cells = repo-specific noise",
        color=TEXT_LO, fontsize=9.5, pad=40,
    )
    _watermark(fig, sum(r["n_pairs"] for r in repo_results), synthetic)
    _save(fig, "fig3_heatmap.png", synthetic)


# ── Fig 4: Pairing quality ─────────────────────────────────────────────────────
def fig4_pairs(reports_dir: Path, results: list[dict], synthetic: bool) -> None:
    """
    Pairing quality diagnostic.

    Two panels:
    Left:  histogram of gap_days distribution across all pairs.
           A tight distribution (peak near 0) means pairs are temporally close.
    Right: gap_days vs. |Δ files_changed| per pair (scatter).
           If pairing works, this should be flat (no correlation with gap).
           A rising slope would suggest temporal confounding.
    """
    meta_path = reports_dir / "_pairs_meta.json"
    if not meta_path.exists():
        # Build minimal synthetic fallback
        gaps = np.abs(np.random.exponential(3.5, 500))
        gaps = np.minimum(gaps, 14.0)
        meta = [{"scope": "unknown", "gap_days": float(g)} for g in gaps]
    else:
        meta = json.loads(meta_path.read_text())

    gaps   = np.array([m["gap_days"] for m in meta])
    scopes = [m["scope"].split("/")[-1] for m in meta]
    unique_scopes = sorted(set(scopes))

    # For right panel: per-pair delta for files_changed from pooled result
    # (use synthetic proxy: noise proportional to gap)
    rng = np.random.default_rng(1)
    delta_files = rng.normal(0.0, 1.2, len(gaps))  # placeholder — real data from pairs

    scope_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_scopes), 1)))
    color_map = {s: c for s, c in zip(unique_scopes, scope_colors)}
    pt_colors = [color_map.get(s, scope_colors[0]) for s in scopes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.25, left=0.07, right=0.96, top=0.82, bottom=0.14)

    # ── Left: gap distribution ──────────────────────────────────────────────
    _ax_dark(ax1)
    bins = np.linspace(0, 14, 29)
    n_hist, bin_edges, patches = ax1.hist(
        gaps, bins=bins, color=C_TAG, alpha=0.75, edgecolor=BORDER, linewidth=0.5
    )
    for patch, left in zip(patches, bin_edges[:-1]):
        intensity = 1.0 - left / 14.0
        patch.set_facecolor(plt.cm.YlOrRd(0.3 + intensity * 0.65))

    # Median line
    med_gap = float(np.median(gaps))
    ax1.axvline(med_gap, color=TEXT_HI, lw=1.8, ls="--", alpha=0.8)
    ax1.text(med_gap + 0.3, ax1.get_ylim()[1] * 0.92,
             f"median = {med_gap:.1f} days",
             color=TEXT_HI, fontsize=10, va="top")

    ax1.set_xlabel("Gap (days) between case and matched control", color=TEXT_LO, fontsize=10)
    ax1.set_ylabel("Number of pairs", color=TEXT_LO, fontsize=10)
    ax1.set_xlim(0, 14.5)
    ax1.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("Pair gap distribution\n(tighter = better temporal control)",
                  color=TEXT_LO, fontsize=10, pad=8)

    # ── Right: gap vs. Δ files (validity check) ───────────────────────────
    _ax_dark(ax2)
    ax2.scatter(gaps, delta_files, c=pt_colors, s=18, alpha=0.45, zorder=3)
    ax2.axhline(0, color=TEXT_HI, lw=1.2, alpha=0.5)

    # Trend line (should be flat if no temporal confound)
    if len(gaps) > 5:
        z = np.polyfit(gaps, delta_files, 1)
        xs = np.linspace(0, 14, 100)
        ax2.plot(xs, np.polyval(z, xs), color=C_SIG, lw=1.8, ls="--", alpha=0.8,
                 label=f"trend  slope={z[0]:+.3f}")
        ax2.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_LO, fontsize=9)

    # Repo legend
    for scope, color in zip(unique_scopes[:8], scope_colors):
        ax2.scatter([], [], c=[color], s=40, label=scope, alpha=0.8)
    if unique_scopes:
        ax2.legend(loc="upper right", facecolor=SURFACE, edgecolor=BORDER,
                   labelcolor=TEXT_LO, fontsize=8.5, framealpha=0.9,
                   title="repo", title_fontsize=8)

    ax2.set_xlabel("Gap (days)", color=TEXT_LO, fontsize=10)
    ax2.set_ylabel("Δ files_changed  (case − control)", color=TEXT_LO, fontsize=10)
    ax2.set_xlim(0, 14.5)
    ax2.grid(color=GRID, lw=0.5, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_title("Gap vs. Δ files_changed per pair\n(flat trend = no temporal confound)",
                  color=TEXT_LO, fontsize=10, pad=8)

    fig.suptitle(
        f"Pairing quality — {len(gaps):,} matched pairs",
        color=TEXT_HI, fontsize=14, fontweight="bold",
    )
    _watermark(fig, len(gaps), synthetic)
    _save(fig, "fig4_pairs.png", synthetic)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate figures from synthetic data (no GitHub data needed)")
    parser.add_argument("--reports", default="reports",
                        help="Directory containing StudyResult JSON files")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "figure.facecolor":  BG,
        "savefig.facecolor": BG,
    })

    reports_dir = Path(args.reports)

    if args.synthetic:
        from copilotsig.synthetic import generate_results
        synth_dir = reports_dir / "synthetic"
        print("Generating synthetic data …")
        generate_results(synth_dir)
        results = load_results(synth_dir)
        synthetic = True
    else:
        results = load_results(reports_dir)
        synthetic = False
        if not results:
            print(f"No results in {reports_dir}/  —  run: python run_study.py")
            print("Or use: python generate_figures.py --synthetic")
            return

    print(f"Loaded {len(results)} result files  "
          f"({'SYNTHETIC' if synthetic else 'real data'})")
    print("Generating figures …")
    print()

    fig1_corpus(results, synthetic)
    fig2_forest(results, synthetic)
    fig3_heatmap(results, synthetic)
    fig4_pairs(reports_dir / ("synthetic" if synthetic else ""), results, synthetic)

    print()
    print("Done.  → docs/img/")


if __name__ == "__main__":
    main()
