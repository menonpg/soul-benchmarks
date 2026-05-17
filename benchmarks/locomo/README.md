# LoCoMo Benchmark

Evaluates soul.py on the [LoCoMo](https://github.com/snap-research/locomo) long-conversation memory benchmark.

## Categories
- **Single-hop**: Questions answerable from a single conversation turn
- **Multi-hop**: Questions requiring reasoning across multiple turns
- **Open-domain**: Open-ended questions scored via LLM-as-judge
- **Temporal**: Questions about when events occurred

## Usage

```bash
python run_locomo.py --config ../../configs/default.yaml
```

Results are saved to `results/` as timestamped JSON files.
