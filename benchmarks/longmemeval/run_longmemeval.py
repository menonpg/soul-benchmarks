"""LongMemEval-S benchmark runner for soul.py.

Reference: https://github.com/xiaowu0162/LongMemEval
Dataset: xiaowu0162/LongMemEval (HuggingFace)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.soul_memory import SoulMemoryAdapter

CATEGORIES = [
    "single-session-assistant",
    "single-session-user",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
    "preference",
]

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    """Load LongMemEval dataset from HuggingFace."""
    from datasets import load_dataset as hf_load

    print("Loading LongMemEval from HuggingFace (xiaowu0162/LongMemEval)...")
    ds = hf_load("xiaowu0162/LongMemEval")
    # Use the test split or first available
    split = "test" if "test" in ds else list(ds.keys())[0]
    return list(ds[split])


def classify_question(item: dict) -> str:
    """Map a LongMemEval item to its category."""
    cat = item.get("category", item.get("type", "")).lower().replace(" ", "-").replace("_", "-")
    for c in CATEGORIES:
        if c.replace("-", "") in cat.replace("-", ""):
            return c
    # Fallback heuristics
    if "temporal" in cat:
        return "temporal-reasoning"
    if "preference" in cat:
        return "preference"
    if "update" in cat:
        return "knowledge-update"
    if "multi" in cat:
        return "multi-session"
    if "assistant" in cat:
        return "single-session-assistant"
    if "user" in cat:
        return "single-session-user"
    return "other"


def score_answer(predicted: str, ground_truths: list[str]) -> float:
    """Score predicted answer against ground truth(s)."""
    pred = predicted.strip().lower()
    best = 0.0
    for gt in ground_truths:
        gt_lower = gt.strip().lower()
        if gt_lower in pred or pred in gt_lower:
            return 1.0
        gt_words = set(gt_lower.split())
        pred_words = set(pred.split())
        if gt_words:
            overlap = len(gt_words & pred_words) / len(gt_words)
            best = max(best, overlap)
    return best


def run_benchmark(config: dict) -> dict:
    """Run the full LongMemEval-S benchmark."""
    dataset = load_dataset()
    modes = config.get("modes", ["auto"])
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running LongMemEval-S benchmark — mode: {mode}")
        print(f"{'='*60}")

        category_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        all_scores: list[float] = []

        adapter = SoulMemoryAdapter(
            provider=config.get("provider", "anthropic"),
            model=config.get("model", "claude-sonnet-4-20250514"),
            mode=mode,
        )

        for i, item in enumerate(dataset):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(dataset)}...")

            # Ingest conversation sessions
            sessions = item.get("sessions", item.get("conversation", item.get("history", [])))
            if isinstance(sessions, list):
                for session in sessions:
                    text = session if isinstance(session, str) else json.dumps(session)
                    adapter.add_memory(text)
            elif isinstance(sessions, str):
                adapter.add_memory(sessions)

            # Query
            question = item.get("question", item.get("query", ""))
            ground_truth = item.get("answer", item.get("answers", ""))
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            category = classify_question(item)
            predicted = adapter.query_memory(question)
            score = score_answer(predicted, ground_truth)

            if category in category_scores:
                category_scores[category].append(score)
            all_scores.append(score)

            time.sleep(0.3)  # Rate limiting

        # Aggregate
        mode_results = {}
        for cat, scores in category_scores.items():
            mode_results[cat] = round(sum(scores) / len(scores), 4) if scores else None
        mode_results["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
        mode_results["total_questions"] = len(all_scores)
        results[mode] = mode_results

        print(f"\nResults for mode={mode}:")
        for k, v in mode_results.items():
            print(f"  {k}: {v}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"longmemeval_results_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval-S benchmark for soul.py")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_benchmark(config)


if __name__ == "__main__":
    main()
