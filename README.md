# soul-benchmarks

Benchmarking [soul.py](https://github.com/menonpg/soul.py) (`pip install soul-agent`) against established long-term memory benchmarks.

soul.py features a **RAG + RLM (Reflective Latent Memory)** hybrid architecture. This repo evaluates it on the same benchmarks used by Mem0, Zep, Xmem, LangMem, and others — enabling direct, apples-to-apples comparison.

## Benchmarks

| Benchmark | Source | Categories |
|-----------|--------|------------|
| **LoCoMo** | [snap-research/locomo](https://github.com/snap-research/locomo) | Single-hop, Multi-hop, Open-domain, Temporal |
| **LongMemEval-S** | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Single-session (assistant/user), Knowledge update, Multi-session, Temporal reasoning, Preference |

## Results Comparison

### LoCoMo

| System | Single-hop | Multi-hop | Open-domain | Temporal | Overall |
|--------|-----------|-----------|-------------|----------|---------|
| Full History (GPT-4) | — | — | — | — | — |
| Mem0 | — | — | — | — | — |
| Zep | — | — | — | — | — |
| Xmem | — | — | — | — | — |
| LangMem | — | — | — | — | — |
| **soul.py (RAG)** | — | — | — | — | — |
| **soul.py (RLM)** | — | — | — | — | — |
| **soul.py (Auto/Hybrid)** | — | — | — | — | — |

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
