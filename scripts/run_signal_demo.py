import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import build_positions, sweep_signal_thresholds

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().parent.parent / "results" / "predictions_lightgbm.csv"
ZSCORE_WINDOW = 20
RAW_THRESHOLDS = [0.0005, 0.001, 0.002]
ZSCORE_THRESHOLDS = [0.5, 1.0, 1.5]
HOLD_DAYS_GRID = [1, 3, 5]
OVERLAP_RULES = ["ignore_new_signal", "restart_clock", "double_down"]

if __name__ == "__main__":
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH
    predictions = pd.read_csv(predictions_path, index_col="date", parse_dates=True)
    oos = predictions.dropna(subset=["y_pred"])
    y_pred = oos["y_pred"]
    y_true = oos["y_true"]

    print(f"Loaded {len(oos)} OOS predictions from {predictions_path}\n")

    grid = sweep_signal_thresholds(y_pred, RAW_THRESHOLDS, ZSCORE_THRESHOLDS, ZSCORE_WINDOW)

    print("=== SIGNAL COUNTS PER THRESHOLD (raw / zscore sweep) ===")
    for (method, threshold), signal in grid.items():
        counts = signal.value_counts().reindex([1, 0, -1], fill_value=0)
        print(f"{method:>7} thr={threshold:<6} long={counts[1]:<5} flat={counts[0]:<5} short={counts[-1]:<5}")
    print()

    print("=== POSITION LOGIC SWEEP (hold_days x overlap_rule), one signal config ===")
    example_signal = grid[("zscore", 1.0)]
    rows = []
    for hold_days in HOLD_DAYS_GRID:
        for overlap_rule in OVERLAP_RULES:
            positions, trades = build_positions(example_signal, hold_days=hold_days, overlap_rule=overlap_rule)
            aligned_true = y_true.reindex(positions.index)
            strategy_ret = positions * aligned_true
            n_trades = sum(1 for t in trades if t.action in ("open", "restart", "flip"))
            n_doubles = sum(1 for t in trades if t.action == "double_down")
            rows.append(
                {
                    "hold_days": hold_days,
                    "overlap_rule": overlap_rule,
                    "n_trades": n_trades,
                    "n_double_downs": n_doubles,
                    "pct_days_in_market": (positions != 0).mean(),
                    "cum_return": strategy_ret.sum(),
                    "mean_daily_ret": strategy_ret.mean(),
                }
            )

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    print()

    print("=== SAMPLE POSITION TRACE (zscore thr=1.0, hold_days=3, restart_clock) ===")
    positions, trades = build_positions(example_signal, hold_days=3, overlap_rule="restart_clock")
    trace = pd.DataFrame(
        {
            "signal": example_signal,
            "position": positions,
            "y_true_next_day_ret": y_true.reindex(positions.index),
        }
    )
    nonzero_start = trace[trace["position"] != 0].index[0]
    print(trace.loc[nonzero_start:].head(15).to_string())
