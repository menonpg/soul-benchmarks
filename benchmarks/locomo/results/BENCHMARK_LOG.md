# LoCoMo Benchmark Run Log

## Date: 2026-05-17/18

### All Results (Final)

| Config | Overall | Single-hop | Multi-hop | Open-domain | Temporal | Questions | Errors |
|--------|---------|-----------|-----------|-------------|----------|-----------|--------|
| **RLM** | **69.99%** | **54.13%** | **82.06%** | 55.10% | **39.99%** | 1986 | 159 |
| Hybrid | 65.57% | 45.97% | 79.49% | 56.04% | 29.84% | 1986 | 149 |
| Auto | 64.05% | 42.56% | 78.46% | 58.75% | 26.72% | 1986 | 33 |
| Qdrant (RAG) | 63.42% | 36.45% | 78.72% | **59.38%** | 26.97% | 1986 | 0 |
| BM25 | 63.05% | 38.40% | 77.80% | 50.83% | 29.26% | 1986 | 0 |

### Key Findings
- **RLM is the clear winner** at 70.0% — 4–7 points above all others
- RLM dominates temporal (+10pts) and single-hop (+8pts)
- Auto (dynamic routing) lands at 64% — between Hybrid and baselines, not as good as always-RLM
- Multi-hop is strong across all configs (78–82%)
- Qdrant wins open-domain (59.4%) — vector search surfaces broad context well
