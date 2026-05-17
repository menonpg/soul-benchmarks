"""Generate markdown comparison tables from benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"

# Published results from competitor papers/repos (fill in as available)
PUBLISHED_LOCOMO = {
    "Mem0": {"single-hop": None, "multi-hop": None, "open-domain": None, "temporal": None, "overall": None},
    "Zep": {"single-hop": None, "multi-hop": None, "open-domain": None, "temporal": None, "overall": None},
    "Xmem": {"single-hop": None, "multi-hop": None, "open-domain": None, "temporal": None, "overall": None},
    "LangMem": {"single-hop": None, "multi-hop": None, "open-domain": None, "temporal": None, "overall": None},
}

PUBLISHED_LONGMEMEVAL = {
    "Mem0": {},
    "Zep": {},
    "Xmem": {},
}


def load_latest_results(benchmark: str) -> dict | None:
    results_dir = BENCHMARKS_DIR / benchmark / "results"
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if isinstance(v, float) else str(v)


def print_locomo_table(results: dict | None):
    headers = ["System", "Single-hop", "Multi-hop", "Open-domain", "Temporal", "Overall"]
    rows = []

    for name, scores in PUBLISHED_LOCOMO.items():
        rows.append([name] + [fmt(scores.get(c)) for c in ["single-hop", "multi-hop", "open-domain", "temporal", "overall"]])

    if results:
        for mode, scores in results.items():
            rows.append([f"soul.py ({mode})"] + [fmt(scores.get(c)) for c in ["single-hop", "multi-hop", "open-domain", "temporal", "overall"]])

    print("\n## LoCoMo Results\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_longmemeval_table(results: dict | None):
    cats = ["single-session-assistant", "single-session-user", "knowledge-update", "multi-session", "temporal-reasoning", "preference", "overall"]
    headers = ["System", "SS-Asst", "SS-User", "Knowl. Update", "Multi-Sess", "Temporal", "Preference", "Overall"]
    rows = []

    for name, scores in PUBLISHED_LONGMEMEVAL.items():
        rows.append([name] + [fmt(scores.get(c)) for c in cats])

    if results:
        for mode, scores in results.items():
            rows.append([f"soul.py ({mode})"] + [fmt(scores.get(c)) for c in cats])

    print("\n## LongMemEval-S Results\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables")
    parser.add_argument("--output", default=None, help="Write to file instead of stdout")
    args = parser.parse_args()

    locomo = load_latest_results("locomo")
    longmemeval = load_latest_results("longmemeval")

    print_locomo_table(locomo)
    print_longmemeval_table(longmemeval)


if __name__ == "__main__":
    main()
