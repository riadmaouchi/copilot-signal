"""
copilot-signal: case-control study of Copilot-tagged commits.

Usage:
    # Discover repos with Copilot-tagged commits, then analyze them
    GITHUB_TOKEN=ghp_... python run_study.py --discover --top 20

    # Analyze specific repos
    GITHUB_TOKEN=ghp_... python run_study.py --repos microsoft/vscode,denoland/deno

    # Tighter pairing window (default: 14 days)
    GITHUB_TOKEN=ghp_... python run_study.py --repos owner/repo --max-gap 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from copilotsig.analyzer.pairing import make_pairs
from copilotsig.analyzer.signals import extract
from copilotsig.analyzer.stats import compare
from copilotsig.collector.cache import Cache
from copilotsig.collector.github import GitHubClient
from copilotsig.models import CommitSignals

_console = Console()
REPORTS = Path("reports")


async def analyze_repo(
    client: GitHubClient,
    repo_full: str,
    max_commits: int,
    max_gap_days: int,
) -> dict:
    owner, _, repo = repo_full.partition("/")
    _console.print(f"\n[cyan]→ {repo_full}[/cyan]")

    signals: list[CommitSignals] = []
    n_tagged = 0

    async for commit in client.iter_repo_commits(owner, repo, max_commits=max_commits):
        sig = extract(commit)
        signals.append(sig)
        if commit.copilot_tagged:
            n_tagged += 1

    total = len(signals)
    _console.print(f"  {total} commits — {n_tagged} Copilot-tagged ({100*n_tagged/max(total,1):.1f}%)")

    if n_tagged == 0:
        _console.print(f"  [yellow]No Copilot-tagged commits found — skipping[/yellow]")
        return {"repo": repo_full, "status": "no_copilot_commits"}

    pairs = make_pairs(signals, max_gap_days=max_gap_days)
    _console.print(f"  {len(pairs)} matched pairs (max_gap={max_gap_days}d)")

    if not pairs:
        _console.print(f"  [yellow]No valid pairs within {max_gap_days}-day window[/yellow]")
        return {"repo": repo_full, "status": "no_pairs"}

    result = compare(pairs, scope=repo_full)
    _console.print(f"  {result.interpretation}")

    # Print signal table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Signal", style="dim")
    table.add_column("Copilot median", justify="right")
    table.add_column("Control median", justify="right")
    table.add_column("Δ%", justify="right")
    table.add_column("p", justify="right")
    table.add_column("r", justify="right")

    for s in result.signals:
        p_str  = f"{s.p_value:.3f}" if s.p_value is not None else "—"
        r_str  = f"{s.effect_size:+.2f}" if s.effect_size is not None else "—"
        delta_str = f"{s.delta_pct:+.1f}%"
        style = "bold green" if s.significant_at_05 else ""
        table.add_row(
            s.signal, str(s.case_median), str(s.control_median),
            delta_str, p_str, r_str,
            style=style,
        )
    _console.print(table)

    # Save
    REPORTS.mkdir(exist_ok=True)
    out_path = REPORTS / f"{repo_full.replace('/', '_')}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    _console.print(f"  [dim]→ {out_path}[/dim]")

    return {"repo": repo_full, "status": "ok", "n_pairs": len(pairs)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", help="Comma-separated list of owner/repo")
    parser.add_argument("--discover", action="store_true",
                        help="Search GitHub for repos with Copilot-tagged commits")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of repos to discover (default: 20)")
    parser.add_argument("--max-commits", type=int, default=1000,
                        help="Max commits to fetch per repo (default: 1000)")
    parser.add_argument("--max-gap", type=int, default=14,
                        help="Max days between case and control (default: 14)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    async with GitHubClient(token=token) as client:

        if args.discover:
            _console.print("[bold]Discovering repos with Copilot-tagged commits...[/bold]")
            repos_info = await client.search_copilot_repos(max_results=args.top)
            if not repos_info:
                _console.print("[red]No repos found.[/red]")
                return
            _console.print(f"Found {len(repos_info)} repos:")
            for r in repos_info[:args.top]:
                _console.print(f"  {r['repo']:50s}  {r['copilot_commits']} tagged commits")
            repo_list = [r["repo"] for r in repos_info[:args.top]]

        elif args.repos:
            repo_list = [r.strip() for r in args.repos.split(",")]

        else:
            parser.print_help()
            sys.exit(0)

        results = []
        for repo_full in repo_list:
            r = await analyze_repo(client, repo_full, args.max_commits, args.max_gap)
            results.append(r)

    # Summary
    _console.rule("[bold]Summary[/bold]")
    ok = [r for r in results if r.get("status") == "ok"]
    total_pairs = sum(r.get("n_pairs", 0) for r in ok)
    _console.print(f"{len(ok)}/{len(results)} repos produced pairs — {total_pairs} total pairs")

    (REPORTS / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
