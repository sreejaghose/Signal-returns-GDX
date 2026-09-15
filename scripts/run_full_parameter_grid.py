import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import DEFAULT_REG_STRENGTH_GRID, run_full_grid

DEFAULT_DATA_PATH = "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOOKBACK_GRID = [10, 20, 30]
MODEL_TYPES = ["elasticnet", "lightgbm"]
RAW_THRESHOLDS = [0.0005, 0.001, 0.002]
ZSCORE_THRESHOLDS = [0.5, 1.0, 1.5]
HOLD_DAYS_GRID = [1, 3, 5, 10]
OVERLAP_RULES = ["ignore_new_signal", "restart_clock", "double_down"]
COST_BPS_GRID = [2, 5, 10]
MIN_TRADES_PER_MONTH = 1.0

RESULTS_PATH = PROJECT_ROOT / "results" / "full_grid_results.pkl"
CACHE_DIR = PROJECT_ROOT / "results" / "backtest_cache"

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    force_rerun = "--force" in sys.argv

    n_expensive = len(LOOKBACK_GRID) * sum(len(DEFAULT_REG_STRENGTH_GRID[m]) for m in MODEL_TYPES)
    n_signal_configs = len(RAW_THRESHOLDS) + len(ZSCORE_THRESHOLDS)
    n_cheap_per_expensive = n_signal_configs * len(HOLD_DAYS_GRID) * len(OVERLAP_RULES) * len(COST_BPS_GRID)
    print(f"Expensive combos (lookback x model_type x reg_strength): {n_expensive}")
    print(f"Cheap combos per expensive combo: {n_cheap_per_expensive}")
    print(f"Total rows expected: {n_expensive * n_cheap_per_expensive}")
    print(f"Results cache: {RESULTS_PATH}")
    print(f"Predictions cache dir: {CACHE_DIR}\n")

    t0 = time.time()
    results = run_full_grid(
        data_path=data_path,
        lookback_grid=LOOKBACK_GRID,
        model_types=MODEL_TYPES,
        reg_strength_grid=DEFAULT_REG_STRENGTH_GRID,
        raw_thresholds=RAW_THRESHOLDS,
        zscore_thresholds=ZSCORE_THRESHOLDS,
        hold_days_grid=HOLD_DAYS_GRID,
        overlap_rules=OVERLAP_RULES,
        cost_bps_grid=COST_BPS_GRID,
        results_path=str(RESULTS_PATH),
        cache_dir=str(CACHE_DIR),
        min_trades_per_month=MIN_TRADES_PER_MONTH,
        force_rerun=force_rerun,
    )
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s. Results shape: {results.shape}\n")

    pd.set_option("display.width", 200)

    n_pass = results["meets_min_frequency"].sum()
    n_fail = (~results["meets_min_frequency"]).sum()
    print(f"meets_min_frequency: {n_pass} pass, {n_fail} flagged as too illiquid\n")

    cols = [
        "lookback", "model_type", "reg_strength", "method", "threshold",
        "hold_days", "overlap_rule", "spread_bps",
        "n_trades", "trades_per_month", "meets_min_frequency",
        "net_sharpe", "net_total_return", "gross_sharpe",
    ]

    print("=== TOP 15 BY NET SHARPE, RESTRICTED TO meets_min_frequency == True ===")
    eligible = results[results["meets_min_frequency"]]
    print(eligible.sort_values("net_sharpe", ascending=False)[cols].head(15).to_string(index=False))
    print()

    print("=== BEST ELIGIBLE COMBO PER MODEL TYPE ===")
    best_per_model = (
        eligible.sort_values("net_sharpe", ascending=False)
        .groupby("model_type")
        .head(1)[cols]
    )
    print(best_per_model.to_string(index=False))
