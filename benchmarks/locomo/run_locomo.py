"""LoCoMo benchmark runner for soul.py.

Reference: https://github.com/snap-research/locomo
Dataset: snap-research/locomo (locomo10.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.soul_memory import SoulMemoryAdapter

CATEGORIES = ["single-hop", "multi-hop", "open-domain", "temporal"]

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(data_path: str | None = None) -> list[dict]:
    """Load LoCoMo dataset from local path or download from GitHub."""
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.load(f)

    # Download from snap-research/locomo
    import httpx

    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo dataset from {url} ...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Cache locally
    cache_path = Path(__file__).parent / "locomo10.json"
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"Cached to {cache_path}")
    return data


def classify_question(question: dict) -> str:
    """Map a LoCoMo question to its category."""
    cat = question.get("category", "").lower()
    for c in CATEGORIES:
        if c.replace("-", "") in cat.replace("-", "").replace("_", ""):
            return c
    return "other"


def score_exact_match(predicted: str, ground_truth: str) -> float:
    """Simple exact/substring match scoring."""
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    if gt in pred or pred in gt:
        return 1.0
    # Check keyword overlap
    gt_words = set(gt.split())
    pred_words = set(pred.split())
    if not gt_words:
        return 0.0
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap


def score_llm_judge(predicted: str, ground_truth: str, question: str, adapter: SoulMemoryAdapter) -> float:
    """Use LLM-as-judge for open-domain questions."""
    prompt = f"""You are a judge evaluating answer quality.

Question: {question}
Ground truth answer: {ground_truth}
Predicted answer: {predicted}

Rate the predicted answer from 0.0 to 1.0 based on correctness and completeness.
Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = adapter.soul.ask(prompt)
        score = float(str(response).strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return score_exact_match(predicted, ground_truth)


def run_benchmark(config: dict, data_path: str | None = None) -> dict:
    """Run the full LoCoMo benchmark."""
    dataset = load_dataset(data_path)
    modes = config.get("modes", ["auto"])
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running LoCoMo benchmark — mode: {mode}")
        print(f"{'='*60}")

        category_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        all_scores: list[float] = []

        for i, conversation in enumerate(dataset):
            print(f"\nConversation {i+1}/{len(dataset)}")
            adapter = SoulMemoryAdapter(
                provider=config.get("provider", "anthropic"),
                model=config.get("model", "claude-sonnet-4-20250514"),
                mode=mode,
            )

            # Ingest conversation sessions
            sessions = conversation.get("conversation", [])
            if isinstance(sessions, list):
                for session in sessions:
                    text = session if isinstance(session, str) else json.dumps(session)
                    adapter.add_memory(text)
            elif isinstance(sessions, str):
                adapter.add_memory(sessions)

            # Answer QA pairs
            questions = conversation.get("questions", conversation.get("qa_pairs", []))
            for q in questions:
                question_text = q.get("question", q.get("q", ""))
                ground_truth = q.get("answer", q.get("a", ""))
                category = classify_question(q)

                predicted = adapter.query_memory(question_text)

                if category == "open-domain":
                    score = score_llm_judge(predicted, ground_truth, question_text, adapter)
                else:
                    score = score_exact_match(predicted, ground_truth)

                if category in category_scores:
                    category_scores[category].append(score)
                all_scores.append(score)

                print(f"  [{category}] score={score:.2f} | Q: {question_text[:60]}...")

            time.sleep(0.5)  # Rate limiting

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
    output_path = RESULTS_DIR / f"locomo_results_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark for soul.py")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data", default=None, help="Path to locomo10.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_benchmark(config, args.data)


if __name__ == "__main__":
    main()
