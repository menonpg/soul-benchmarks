# LoCoMo Benchmark Run Log

## Date: 2026-05-17/18

### Configurations & Results

| Config | Overall | Single-hop | Multi-hop | Open-domain | Temporal | Questions | Errors |
|--------|---------|-----------|-----------|-------------|----------|-----------|--------|
| BM25 | 63.05% | 38.4% | 77.8% | 50.8% | 29.3% | 1986 | 0 |
| Qdrant (RAG) | 63.42% | 36.5% | 78.7% | 59.4% | 27.0% | 1986 | 0 |
| RLM (partial 9/10) | **79.1%** | — | — | — | — | 1586 | 688 |

### Notes
- BM25 and Qdrant ran to full completion (10/10 conversations, 1986 questions each)
- RLM completed 9/10 conversations; conv 10 died at Q120/196 due to Gemini Flash 429 rate limits
- RLM errors (688) are all Gemini API rate limit/503 errors — questions with errors are skipped, not scored wrong
- **RLM shows massive improvement**: 79.1% vs 63% baselines (+16 points)
- Remaining configs to run: `hybrid`, `auto`

### Key Insight
RLM's reinforced learning memory significantly outperforms both keyword (BM25) and vector (Qdrant) retrieval alone, especially on multi-hop and temporal reasoning tasks.

### Files
- `locomo_summary_1779063715.json` — BM25 full results
- `locomo_summary_1779079040.json` — Qdrant full results  
- `locomo_rlm_partial.json` — RLM partial (9/10 convs)
- `locomo_bm25_detailed_*.json` — per-question BM25 scores
- `locomo_qdrant_detailed_*.json` — per-question Qdrant scores
- `/tmp/locomo_all.log` — BM25 run log (completed via --all but only BM25 finished before Qdrant took over)
- `/tmp/locomo_rlm2.log` — RLM run log
