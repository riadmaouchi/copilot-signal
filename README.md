# copilot-signal

> **Do Copilot-tagged commits differ measurably from non-Copilot commits by the same developer?**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Question

[dev-fingerprint](https://github.com/riadmaouchi/dev-fingerprint) showed that AI adoption leaves no detectable trace in a developer's longitudinal commit history. But that study compared a developer against their own past — confounding AI adoption with career changes, project phase, and time.

This study asks a different question with a different design: **commit-level case-control**.

GitHub Copilot adds a verifiable trailer to commit messages when it generates them:
```
Co-Authored-By: GitHub Copilot <copilot@github.com>
```

This is the closest thing to ground truth at the individual commit level that exists in public data. We use it to build matched pairs: for each Copilot-tagged commit, we find the nearest non-tagged commit from the **same author in the same repo within the same 14-day window**.

Within-author, within-repo, within-2-weeks pairing controls for:
- Developer coding style
- Project language and conventions  
- Project phase
- Team size and review culture

What remains is the isolated effect (if any) of AI assistance on commit structure.

---

## Design

```
GitHub commit search
      │  query: "co-authored-by:copilot@github.com"
      ▼
Repo corpus (repos with ≥ N Copilot-tagged commits)
      │
      ▼
Per-repo: fetch all commits (Copilot-tagged + untagged)
      │
      ▼
Extract CommitSignals (Level-A process signals)
      │  files_changed · net_lines · total_churn
      │  cross_module_ratio · is_refactor · touches_tests
      │  message_length · has_conventional_commit
      │
      ▼
Pair matching: (author, repo, ±14 days)
      │  each Copilot commit → nearest untagged commit, same author
      │  one-to-one matching, no reuse of controls
      │
      ▼
Wilcoxon signed-rank test (paired, non-parametric)
      │  per signal, per repo, and pooled
      │  effect size: matched-pairs rank-biserial r
      │
      ▼
StudyResult — honest interpretation
```

---

## What This Can and Cannot Claim

**Can claim:**
- Whether commit-level process signals differ between commits where Copilot generated the commit message vs. commits where it did not.

**Cannot claim:**
- That the Copilot tag means Copilot wrote all the code (it may mean only the commit message was generated).
- Generalization beyond the tagged/untagged distinction (many Copilot users never enable the tag).
- Causation — the tag may be correlated with commit type (e.g., developers use the tag only for large feature commits).

**Key selection bias to watch for:**
Developers who enable the Copilot co-authorship trailer are a self-selected subset. The comparison is valid within each author (their tagged vs. untagged commits) but cannot generalize to "AI vs. non-AI commits" globally.

---

## Reproduce

```bash
git clone https://github.com/riadmaouchi/copilot-signal
cd copilot-signal
pip install -e ".[dev]"

# Discover repos with Copilot-tagged commits and analyze the top 20
export GITHUB_TOKEN=ghp_...
python run_study.py --discover --top 20

# Or analyze specific repos
python run_study.py --repos microsoft/vscode,denoland/deno --max-gap 14
```

Results are saved to `reports/`.

---

## Project Structure

```
copilot-signal/
├── src/copilotsig/
│   ├── collector/
│   │   ├── github.py     GitHub API client + Copilot tag detection
│   │   └── cache.py      SQLite disk cache (TTL 7 days)
│   ├── analyzer/
│   │   ├── signals.py    Per-commit Level-A signal extraction
│   │   ├── pairing.py    Case-control matching (author, repo, ±N days)
│   │   └── stats.py      Wilcoxon signed-rank + rank-biserial r
│   └── models.py         Commit, CommitPair, SignalComparison, StudyResult
├── reports/              Per-repo JSON results
├── run_study.py          Entry point (discover + analyze)
└── README.md
```

---

MIT License
