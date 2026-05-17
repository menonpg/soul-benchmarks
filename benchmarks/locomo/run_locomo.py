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
from adapters.soul_memory import SoulMemoryAdapter, create_gemini_adapter

CATEGORIES = ["single-hop", "multi-hop", "open-domain", "temporal"]

CATEGORY_MAP = {
    1: "single-hop",
    2: "temporal",
    3: "open-domain",
    4: "multi-hop",
    5: "multi-hop",
}

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(data_path: str | None = None) -> list[dict]:
    """Load LoCoMo dataset from local path or download from GitHub."""
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.load(f)

    import httpx

    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo dataset from {url} ...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    cache_path = Path(__file__).parent / "locomo10.json"
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"Cached to {cache_path}")
    return data


def classify_question(question: dict) -> str:
    cat = question.get("category")
    if isinstance(cat, int):
        return CATEGORY_MAP.get(cat, "other")
    return "other"


def score_exact_match(predicted: str, ground_truth) -> float:
    pred = str(predicted).strip().lower()
    gt = str(ground_truth).strip().lower()
    if gt in pred or pred in gt:
        return 1.0
    gt_words = set(gt.split())
    pred_words = set(pred.split())
    if not gt_words:
        return 0.0
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap


def score_llm_judge(predicted: str, ground_truth, question: str, judge_adapter: SoulMemoryAdapter) -> float:
    prompt = f"""You are a judge evaluating answer quality. Rate from 0.0 to 1.0.

Question: {question}
Ground truth answer: {ground_truth}
Predicted answer: {predicted}

Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = judge_adapter.query_memory(prompt)
        # Extract first float from response
        import re
        match = re.search(r'(\d+\.?\d*)', str(response))
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return score_exact_match(predicted, ground_truth)
    except Exception:
        return score_exact_match(predicted, ground_truth)


def make_adapter(config: dict, mode: str) -> SoulMemoryAdapter:
    return create_gemini_adapter(mode=mode)


def format_session(session: list, timestamp: str = "") -> str:
    """Format a LoCoMo session (list of dialog turns) into text."""
    parts = []
    if timestamp:
        parts.append(f"[Date: {timestamp}]")
    for turn in session:
        if isinstance(turn, dict):
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if speaker and text:
                parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def run_benchmark(config: dict, data_path: str | None = None) -> dict:
    """Run the full LoCoMo benchmark."""
    dataset = load_dataset(data_path)
    modes = config.get("modes", ["auto"])
    results = {}

    # Create a judge adapter (separate from the test adapter)
    judge = create_gemini_adapter(mode="rag")

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running LoCoMo benchmark — mode: {mode}")
        print(f"{'='*60}")

        category_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        all_scores: list[float] = []
        detailed_results: list[dict] = []

        for i, conversation in enumerate(dataset):
            print(f"\nConversation {i+1}/{len(dataset)} (sample: {conversation.get('sample_id', '?')})")
            adapter = make_adapter(config, mode)

            # Ingest all sessions
            conv_data = conversation.get("conversation", {})
            session_keys = sorted(
                [k for k in conv_data.keys() if k.startswith("session_") and not k.endswith("_date_time")],
                key=lambda x: int(x.split("_")[1])
            )

            for skey in session_keys:
                session = conv_data[skey]
                ts_key = f"{skey}_date_time"
                timestamp = conv_data.get(ts_key, "")
                text = format_session(session, timestamp)
                try:
                    adapter.add_memory(text)
                    print(f"  Ingested {skey} ({len(session)} turns)")
                except Exception as e:
                    print(f"  ERROR ingesting {skey}: {e}")
                time.sleep(0.3)

            # Answer QA pairs
            questions = conversation.get("qa", [])
            for qi, q in enumerate(questions):
                question_text = q.get("question", "")
                ground_truth = str(q.get("answer", q.get("a", "")))
                category = classify_question(q)

                try:
                    predicted = adapter.query_memory(question_text)
                except Exception as e:
                    print(f"  ERROR on Q{qi}: {e}")
                    predicted = ""
                    time.sleep(2)

                # Score
                if category == "open-domain":
                    score = score_llm_judge(predicted, ground_truth, question_text, judge)
                else:
                    score = score_exact_match(predicted, ground_truth)

                if category in category_scores:
                    category_scores[category].append(score)
                all_scores.append(score)

                detailed_results.append({
                    "conversation": i,
                    "question": question_text,
                    "ground_truth": ground_truth,
                    "predicted": predicted[:500],
                    "category": category,
                    "score": score,
                })

                if qi % 10 == 0:
                    print(f"  Q{qi}/{len(questions)} [{category}] score={score:.2f}")

                time.sleep(0.3)

            # Print running totals
            current_total = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
            print(f"  Running overall: {current_total:.1f}% ({len(all_scores)} questions)")

        # Aggregate
        mode_results = {}
        for cat, scores in category_scores.items():
            pct = round(sum(scores) / len(scores) * 100, 2) if scores else None
            mode_results[cat] = pct
        overall = round(sum(all_scores) / len(all_scores) * 100, 2) if all_scores else None
        mode_results["overall"] = overall
        mode_results["total_questions"] = len(all_scores)
        results[mode] = mode_results

        print(f"\n{'='*60}")
        print(f"RESULTS — mode: {mode}")
        print(f"{'='*60}")
        for k, v in mode_results.items():
            if k != "total_questions":
                print(f"  {k}: {v}%")
            else:
                print(f"  {k}: {v}")

        # Save detailed results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        detail_path = RESULTS_DIR / f"locomo_{mode}_detailed_{ts}.json"
        with open(detail_path, "w") as f:
            json.dump(detailed_results, f, indent=2)

    # Save summary results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_path = RESULTS_DIR / f"locomo_results_{ts}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark for soul.py")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data", default=None, help="Path to locomo10.json")
    parser.add_argument("--mode", default=None, help="Override mode: rag, rlm, or auto")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["modes"] = [args.mode]

    run_benchmark(config, args.data)


if __name__ == "__main__":
    main()
