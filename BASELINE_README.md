# Baseline Backtest (Phase 1)

This baseline checks buy-and-hold and simple momentum rules (ret_5, ret_21) using the Phase 1 data spec.

## Usage

```bash
python baseline_backtest.py --raw-path data/SPY/raw/spy_raw.parquet --out-dir data/SPY/baseline --snapshot-id s001
```

## Output
- JSON report saved to the output directory.
- Printed summary to stdout.

## Notes
- Uses shift-safe feature computation from pipeline_spec.py.
- Execution assumes signal at date T, entry at next open, exit at next close.
