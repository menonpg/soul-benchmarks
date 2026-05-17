# JOURNAL.md — soul-benchmarks

Full record of the benchmarking effort for soul.py against LoCoMo and LongMemEval-S.

## Origin Story (2026-05-17)

Ishaan Gupta (XortexAI / Newton School) emailed Prahlad about **Xmem** — their open-source memory layer for AI agents. They published benchmarks on **LoCoMo** (ACL 2024) and **LongMemEval-S** (ICLR 2025) showing competitive results against Mem0, Zep, and others.

This sparked the idea: benchmark **soul.py** (our RAG+RLM hybrid memory package) on the same benchmarks.

## What We're Benchmarking

**soul.py** (`pip install soul-agent`) is a persistent memory system for AI agents. Its key innovation is the **RAG+RLM hybrid**:
- **RAG** (Retrieval-Augmented Generation): Store memories as vectors, retrieve by semantic similarity
- **RLM** (Reflexive Long-term Memory): LLM recursively reasons over stored memories
- **Auto router**: Classifies queries and picks the best retrieval strategy

### Benchmarks

**LoCoMo** — 10 long-term conversations, ~2,000 QA pairs
- Single-hop: one fact, one answer (282 Qs)
- Multi-hop: connect 2+ facts (1,287 Qs)
- Temporal: date/time reasoning (321 Qs)
- Open-domain: reasoning from multiple memories (96 Qs)
- Source: https://github.com/snap-research/locomo

**LongMemEval-S** — ~500 questions across 6 categories
- Source: https://github.com/xiaowu0162/LongMemEval

### Configurations

| # | Config | RAG Backend | RLM | Purpose |
|---|--------|------------|-----|---------|
| 1 | BM25 only | BM25 keyword | ❌ | No-ML baseline |
| 2 | Qdrant only | Qdrant + embeddings | ❌ | Semantic search alone |
| 3 | RLM only | ❌ | ✅ | Pure LLM reasoning |
| 4 | Qdrant + RLM | Qdrant + embeddings | ✅ | Full hybrid (flagship) |
| 5 | Auto | Router picks | ✅ | Real-world mode |

### Infrastructure

- **LLM:** Gemini 2.0 Flash (free tier) — same family Xmem used for their benchmarks
- **Embeddings:** Azure text-embedding-3-large (3072 dims)
- **Vector DB:** Qdrant Cloud (existing instance, NEW collections with `locomo_bench_` prefix)
- **Scoring:** Exact match for factual Qs, LLM-as-judge for open-domain

## Comparison Targets

Published numbers from competitors on LoCoMo:

| Method | Single-Hop | Multi-Hop | Open Domain | Temporal | Overall |
|--------|-----------|-----------|-------------|----------|---------|
| XMem (Gemini 3-flash) | 90.6 | 92.3 | 91.2 | 91.9 | 91.5% |
| Zep | 74.11 | 66.04 | 67.71 | 79.79 | 75.14% |
| Memobase | 70.92 | 46.88 | 77.17 | 85.05 | 75.78% |
| Mem0g (YC 24) | 65.71 | 47.19 | 75.71 | 58.13 | 68.44% |
| Mem0 (YC 24) | 67.13 | 51.15 | 72.93 | 55.51 | 66.88% |
| LangMem | 62.23 | 47.92 | 71.12 | 23.43 | 58.10% |
| OpenAI | 63.79 | 42.92 | 62.29 | 21.71 | 52.90% |

## Code Structure

```
soul-benchmarks/
├── adapters/
│   └── soul_memory.py      # Wraps HybridAgent from soul.py repo
├── benchmarks/
│   ├── locomo/
│   │   ├── run_locomo.py    # Main benchmark runner
│   │   ├── locomo10.json    # Cached dataset (2MB)
│   │   └── results/         # Output JSONs
│   └── longmemeval/
│       └── run_longmemeval.py
├── configs/
│   └── default.yaml
├── scripts/
│   └── compare.py
├── JOURNAL.md               # This file
└── README.md
```

## Key Code: Adapter (adapters/soul_memory.py)

The adapter imports `HybridAgent` directly from the local soul.py repo (not the pip package, which is v0.2.1 and lacks the hybrid features). It:
- Creates a temp dir with SOUL.md + MEMORY.md
- Supports mode switching (rag/rlm/auto)
- `add_memory()` → `agent.remember()`
- `query_memory()` → `agent.ask()`

## Bug Fixes Applied

1. **RLM model hardcoding** — `rlm_memory.py` defaulted to `claude-haiku-4-5` even when using Gemini. Fixed `hybrid_agent.py` to pass `self.chat_model` to RLMMemory's `sub_model` and `synth_model`.

2. **Ground truth type casting** — Some LoCoMo answers are integers (e.g., year "2022"). Added `str()` conversion in scoring functions.

## Credentials (DO NOT COMMIT THESE)

All credentials stored in `~/clawd/secrets/`:
- `azure-openai.env` — Azure OpenAI, Gemini API key
- `api_keys.json` — Qdrant Cloud URL + API key

⚠️ **Qdrant safety:** 48 existing collections on the instance. Benchmark uses ONLY new collections with `locomo_bench_` prefix. Never touch existing collections.

## Timeline

- **2026-05-17 09:00** — Xmem email received, decided to benchmark
- **2026-05-17 09:06** — Created soul-benchmarks repo scaffold
- **2026-05-17 13:15** — fengtrace discussion on soul.py#4 (confidence scoring)
- **2026-05-17 14:41** — Adapter tested with Azure OpenAI (GPT-5) — works
- **2026-05-17 17:06** — New Gemini key obtained, all 3 modes (rag/rlm/auto) tested ✅
- **2026-05-17 18:43** — First benchmark attempt (BM25/RAG). Partial results: low scores expected without semantic retrieval
- **2026-05-17 18:48** — Decision: run 5 configs (BM25, Qdrant, RLM, Qdrant+RLM, Auto) with Qdrant for proper semantic search
- **2026-05-17 18:51** — Documenting everything before proceeding

## Publication Plan

If results are competitive:
1. **arXiv paper** — "soul.py: Lightweight RAG+RLM Hybrid Memory for AI Agents"
2. **HuggingFace** — Publish benchmark results dataset
3. **soul.py README** — Add comparison table
4. **Blog post** — Detailed writeup on blog.themenonlab.com
