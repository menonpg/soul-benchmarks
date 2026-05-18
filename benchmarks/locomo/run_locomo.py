"""LoCoMo benchmark runner for soul.py.

Reference: https://github.com/snap-research/locomo
Dataset: snap-research/locomo (locomo10.json)

Runs all 5 configurations:
  1. bm25   — BM25 keyword search only
  2. qdrant — Qdrant semantic search only
  3. rlm    — RLM reasoning only
  4. hybrid — Qdrant + RLM (flagship)
  5. auto   — Router picks per query
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.soul_memory import CONFIGS, SoulMemoryAdapter, create_bm25_adapter

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
    cache_path = Path(__file__).parent / "locomo10.json"
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.load(f)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import httpx
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo dataset from {url} ...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    data = resp.json()
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


def score_llm_judge(predicted: str, ground_truth, question: str, judge: SoulMemoryAdapter) -> float:
    prompt = f"""You are a judge. Rate the predicted answer from 0.0 to 1.0.

Question: {question}
Ground truth: {ground_truth}
Predicted: {predicted}

Reply with ONLY a number between 0.0 and 1.0."""

    try:
        response = judge.query_memory(prompt)
        match = re.search(r'(\d+\.?\d*)', str(response))
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return score_exact_match(predicted, ground_truth)
    except Exception:
        return score_exact_match(predicted, ground_truth)


def format_session(session: list, timestamp: str = "") -> str:
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


def run_single_config(config_name: str, dataset: list, judge: SoulMemoryAdapter) -> dict:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*60}")

    factory = CONFIGS[config_name]
    category_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    all_scores: list[float] = []
    detailed_results: list[dict] = []
    errors = 0

    for i, conversation in enumerate(dataset):
        print(f"\n  Conversation {i+1}/{len(dataset)} (sample: {conversation.get('sample_id', '?')})")

        # Create fresh adapter per conversation
        try:
            if config_name in ("qdrant", "hybrid", "auto"):
                adapter = factory(collection_suffix=f"_c{i}")
            else:
                adapter = factory()
        except Exception as e:
            print(f"  ERROR creating adapter: {e}")
            errors += 1
            continue

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
            except Exception as e:
                print(f"    ERROR ingesting {skey}: {e}")
                time.sleep(1)
            time.sleep(0.3)

        print(f"    Ingested {len(session_keys)} sessions")

        # Answer QA pairs
        questions = conversation.get("qa", [])
        conv_scores = []
        for qi, q in enumerate(questions):
            question_text = q.get("question", "")
            ground_truth = str(q.get("answer", ""))
            category = classify_question(q)

            try:
                predicted = adapter.query_memory(question_text)
            except Exception as e:
                predicted = ""
                errors += 1
                if errors % 5 == 0:
                    print(f"    ERROR #{errors}: {e}")
                time.sleep(2)

            if category == "open-domain":
                score = score_llm_judge(predicted, ground_truth, question_text, judge)
            else:
                score = score_exact_match(predicted, ground_truth)

            if category in category_scores:
                category_scores[category].append(score)
            all_scores.append(score)
            conv_scores.append(score)

            detailed_results.append({
                "config": config_name,
                "conversation": i,
                "question_idx": qi,
                "question": question_text,
                "ground_truth": ground_truth,
                "predicted": predicted[:500],
                "category": category,
                "score": score,
            })

            if qi % 20 == 0:
                print(f"    Q{qi}/{len(questions)} [{category}] score={score:.2f}")

            time.sleep(0.3)

        conv_avg = sum(conv_scores) / len(conv_scores) * 100 if conv_scores else 0
        print(f"    Conv {i+1} avg: {conv_avg:.1f}% ({len(conv_scores)} Qs)")

        # Cleanup Qdrant collection
        if hasattr(adapter, 'cleanup_collection'):
            adapter.cleanup_collection()

        running_avg = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        print(f"    Running overall: {running_avg:.1f}% ({len(all_scores)} Qs, {errors} errors)")

    # Aggregate
    results = {"config": config_name}
    for cat, scores in category_scores.items():
        pct = round(sum(scores) / len(scores) * 100, 2) if scores else None
        results[cat] = pct
    results["overall"] = round(sum(all_scores) / len(all_scores) * 100, 2) if all_scores else None
    results["total_questions"] = len(all_scores)
    results["errors"] = errors

    print(f"\n{'='*60}")
    print(f"RESULTS — {config_name}")
    print(f"{'='*60}")
    for k, v in results.items():
        if k not in ("config", "total_questions", "errors"):
            print(f"  {k}: {v}%")
    print(f"  total_questions: {results['total_questions']}")
    print(f"  errors: {results['errors']}")

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    detail_path = RESULTS_DIR / f"locomo_{config_name}_detailed_{ts}.json"
    with open(detail_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"  Saved: {detail_path}")

    return results


def run_benchmark(configs: list[str], data_path: str | None = None) -> dict:
    """Run benchmark for multiple configurations."""
    dataset = load_dataset(data_path)
    judge = create_bm25_adapter()  # Lightweight judge

    all_results = {}
    for config_name in configs:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
            continue
        result = run_single_config(config_name, dataset, judge)
        all_results[config_name] = result

        # Save running summary after each config
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        summary_path = RESULTS_DIR / f"locomo_summary_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"{'Config':<12} {'Single-Hop':>10} {'Multi-Hop':>10} {'Open-Dom':>10} {'Temporal':>10} {'Overall':>10}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        sh = f"{r.get('single-hop', '—')}%" if r.get('single-hop') is not None else "—"
        mh = f"{r.get('multi-hop', '—')}%" if r.get('multi-hop') is not None else "—"
        od = f"{r.get('open-domain', '—')}%" if r.get('open-domain') is not None else "—"
        tp = f"{r.get('temporal', '—')}%" if r.get('temporal') is not None else "—"
        ov = f"{r.get('overall', '—')}%" if r.get('overall') is not None else "—"
        print(f"{name:<12} {sh:>10} {mh:>10} {od:>10} {tp:>10} {ov:>10}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark for soul.py")
    parser.add_argument("--config", default=None, help="Run specific config: bm25, qdrant, rlm, hybrid, auto")
    parser.add_argument("--all", action="store_true", help="Run all 5 configurations")
    parser.add_argument("--data", default=None, help="Path to locomo10.json")
    args = parser.parse_args()

    if args.all:
        configs = ["bm25", "qdrant", "rlm", "hybrid", "auto"]
    elif args.config:
        configs = [args.config]
    else:
        configs = ["bm25"]  # Default to cheapest

    run_benchmark(configs, args.data)


if __name__ == "__main__":
    main()
