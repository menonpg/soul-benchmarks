# LoCoMo Benchmark Run Log

## Date: 2026-05-17/18 (Original) + 2026-05-19/20 (Graph Modes)

### Model Configuration
- **Base LLM:** Gemini 2.0 Flash (generation, routing, RLM synthesis, LLM judge, graph extraction)
- **Embeddings:** Azure OpenAI text-embedding-3-large (3072-dim, for Qdrant-based configs)
- **Vector DB:** Qdrant Cloud (Cosine similarity)
- **⚠️ Results are model-dependent** — different LLMs will produce different scores

### All Results (Final)

| Config | Description | Overall | Single-hop | Multi-hop | Open-domain | Temporal | Questions | Errors |
|--------|-------------|---------|-----------|-----------|-------------|----------|-----------|--------|
| **RLM** | Recursive LLM synthesis — processes all memory chunks via recursive summarization | **69.99%** | **54.13%** | **82.06%** | 55.10% | **39.99%** | 1986 | 159 |
| RAG+Graph | Qdrant semantic search + LLM entity-graph extraction and traversal | 67.36% | 45.70% | 81.37% | 52.34% | 34.70% | 1986 | 500 |
| Hybrid | Qdrant semantic RAG + RLM combined (flagship) | 65.57% | 45.97% | 79.49% | 56.04% | 29.84% | 1986 | 149 |
| Auto | Router dynamically picks RAG or RLM per query | 64.05% | 42.56% | 78.46% | 58.75% | 26.72% | 1986 | 33 |
| Qdrant (RAG) | Qdrant vector semantic search only | 63.42% | 36.45% | 78.72% | **59.38%** | 26.97% | 1986 | 0 |
| BM25 | BM25 keyword search only — zero ML baseline | 63.05% | 38.40% | 77.80% | 50.83% | 29.26% | 1986 | 0 |
| Graph Only | LLM entity-graph extraction + graph traversal only (no RAG/RLM) | 38.26% | 16.53% | 50.80% | 8.33% | 16.01% | 1986 | 0 |

### Key Findings
- **RLM is the clear winner** at 70.0% — 4–7 points above all others
- **RAG+Graph** is the #2 performer at 67.4% — graph context helps multi-hop significantly
- RLM dominates temporal (+10pts) and single-hop (+8pts)
- RAG+Graph had high error rate (500) — LLM extraction adds fragility
- Auto (dynamic routing) lands at 64% — between Hybrid and baselines
- Multi-hop is strong across all configs (78–82%)
- Qdrant wins open-domain (59.4%) — vector search surfaces broad context well
- **Graph Only is insufficient** (38.3%) — graph needs to be combined with other retrieval
