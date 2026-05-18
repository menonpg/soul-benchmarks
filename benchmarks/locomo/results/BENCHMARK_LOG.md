# LoCoMo Benchmark Run Log

## Date: 2026-05-17/18

### All Results

| Config | Overall | Single-hop | Multi-hop | Open-domain | Temporal | Questions | Errors |
|--------|---------|-----------|-----------|-------------|----------|-----------|--------|
| BM25 | 63.05% | 38.4% | 77.8% | 50.8% | 29.3% | 1986 | 0 |
| Qdrant (RAG) | 63.42% | 36.5% | 78.7% | 59.4% | 27.0% | 1986 | 0 |
| Hybrid (BM25+RAG) | 65.57% | 46.0% | 79.5% | 56.0% | 29.8% | 1986 | 149 |
| **RLM** | **69.99%** | **54.1%** | **82.1%** | 55.1% | **40.0%** | 1986 | 159 |
| Auto | _running_ | — | — | — | — | — | — |

### Key Findings
- **RLM is the clear winner** at 70.0% overall — 7 points above baselines
- RLM dominates on **temporal** (40% vs ~28-30% for others) and **single-hop** (54% vs 36-46%)
- Multi-hop is strong across all configs (78-82%)
- Hybrid provides modest lift over pure BM25/Qdrant but well below RLM
- Errors in RLM/Hybrid are Gemini API rate limits (429s), not scoring failures

### Files
- `locomo_summary_1779063715.json` — BM25
- `locomo_summary_1779079040.json` — Qdrant  
- `locomo_summary_1779135584.json` — Hybrid
- `locomo_summary_1779137909.json` — RLM
