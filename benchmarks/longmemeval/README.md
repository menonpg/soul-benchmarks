# LongMemEval-S Benchmark

Evaluates soul.py on the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark (short variant).

## Categories
- **Single-session assistant/user**: Recalling info from a single session
- **Knowledge update**: Handling updated/corrected information
- **Multi-session**: Reasoning across multiple sessions
- **Temporal reasoning**: Time-aware memory queries
- **Preference**: Tracking user preferences over time

## Usage

```bash
python run_longmemeval.py --config ../../configs/default.yaml
```

Results are saved to `results/` as timestamped JSON files.
