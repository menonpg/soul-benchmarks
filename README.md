# soul-benchmarks

Benchmarking [soul.py](https://github.com/menonpg/soul.py) (`pip install soul-agent`) against established long-term memory benchmarks.

soul.py features a **RAG + RLM (Reflective Latent Memory)** hybrid architecture. This repo evaluates it on the same benchmarks used by Mem0, Zep, Xmem, LangMem, and others — enabling direct, apples-to-apples comparison.

## Benchmarks

| Benchmark | Source | Categories |
|-----------|--------|------------|
| **LoCoMo** | [snap-research/locomo](https://github.com/snap-research/locomo) | Single-hop, Multi-hop, Open-domain, Temporal |
| **LongMemEval-S** | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Single-session (assistant/user), Knowledge update, Multi-session, Temporal reasoning, Preference |

## Results Comparison

### LoCoMo (1,986 questions across 10 conversations)

All soul.py configs use **Gemini 2.0 Flash** as the LLM. Scored via exact match (factual) and LLM judge (open-domain).

| System | Description | Single-hop | Multi-hop | Open-domain | Temporal | Overall |
|--------|-------------|-----------|-----------|-------------|----------|---------|
| **soul.py (RLM)** | Recursive LLM synthesis — processes all memory chunks | **54.13%** | **82.06%** | 55.10% | **39.99%** | **69.99%** |
| **soul.py (RAG+Graph)** | Qdrant semantic search + LLM entity-graph extraction | 45.70% | 81.37% | 52.34% | 34.70% | 67.36% |
| **soul.py (Hybrid)** | Qdrant semantic RAG + RLM combined | 45.97% | 79.49% | 56.04% | 29.84% | 65.57% |
| **soul.py (Auto)** | Router dynamically picks RAG or RLM per query | 42.56% | 78.46% | 58.75% | 26.72% | 64.05% |
| **soul.py (Qdrant RAG)** | Qdrant vector semantic search only | 36.45% | 78.72% | **59.38%** | 26.97% | 63.42% |
| **soul.py (BM25)** | BM25 keyword search — zero ML baseline | 38.40% | 77.80% | 50.83% | 29.26% | 63.05% |
| **soul.py (Graph Only)** | LLM entity-graph extraction + traversal only | 16.53% | 50.80% | 8.33% | 16.01% | 38.26% |
| Mem0 | — | — | — | — | — | — |
| Zep | — | — | — | — | — | — |
| Xmem | — | — | — | — | — | — |
| LangMem | — | — | — | — | — | — |

**Key Findings:**
- **RLM is the clear winner** at 70.0% — exhaustive recursive synthesis beats selective retrieval
- **RAG+Graph** (#2 at 67.4%) shows knowledge graphs improve multi-hop reasoning
- Graph Only (38.3%) confirms graphs must be combined with other retrieval — not standalone
- Qdrant RAG wins open-domain (59.4%) — vector search surfaces broad context well
- Multi-hop is consistently strong (78–82%) across all retrieval-based configs

### LongMemEval-S

| System | SS-Asst | SS-User | Knowledge Update | Multi-Session | Temporal | Preference | Overall |
|--------|---------|---------|-----------------|---------------|----------|------------|---------|
| Mem0 | — | — | — | — | — | — | — |
| Zep | — | — | — | — | — | — | — |
| Xmem | — | — | — | — | — | — | — |
| **soul.py (RAG)** | — | — | — | — | — | — | — |
| **soul.py (RLM)** | — | — | — | — | — | — | — |
| **soul.py (Auto/Hybrid)** | — | — | — | — | — | — | — |

## Quick Start

```bash
pip install -r requirements.txt

# Run LoCoMo benchmark
python benchmarks/locomo/run_locomo.py --config configs/default.yaml

# Run LongMemEval benchmark
python benchmarks/longmemeval/run_longmemeval.py --config configs/default.yaml

# Generate comparison tables
python scripts/compare.py
```

## Configuration

Edit `configs/default.yaml` to set your LLM provider, model, and memory modes to test.

## Project Structure

```
soul-benchmarks/
├── README.md
├── benchmarks/
│   ├── locomo/          # LoCoMo benchmark runner
│   └── longmemeval/     # LongMemEval-S benchmark runner
├── adapters/
│   └── soul_memory.py   # soul.py memory adapter
├── configs/
│   └── default.yaml     # Default configuration
├── scripts/
│   └── compare.py       # Results comparison table generator
└── requirements.txt
```

## License

MIT
