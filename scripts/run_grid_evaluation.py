import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import CostModel, evaluate_signal_grid

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().parent.parent / "results" / "predictions_lightgbm.csv"
RAW_THRESHOLDS = [0.0005, 0.001, 0.002, 0.003]
ZSCORE_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
ZSCORE_WINDOW = 20
HOLD_DAYS_GRID = [1, 3, 5, 10]
OVERLAP_RULES = ["ignore_new_signal", "restart_clock", "double_down"]
SPREAD_BPS = 5
MIN_TRADES_PER_MONTH = 1.0

if __name__ == "__main__":
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH
    predictions = pd.read_csv(predictions_path, index_col="date", parse_dates=True)
    oos = predictions.dropna(subset=["y_pred"])
    y_pred, y_true = oos["y_pred"], oos["y_true"]

    results = evaluate_signal_grid(
        y_pred,
        y_true,
        raw_thresholds=RAW_THRESHOLDS,
        zscore_thresholds=ZSCORE_THRESHOLDS,
        zscore_window=ZSCORE_WINDOW,
        hold_days_grid=HOLD_DAYS_GRID,
        cost_model=CostModel(spread_bps=SPREAD_BPS),
        overlap_rules=OVERLAP_RULES,
        min_trades_per_month=MIN_TRADES_PER_MONTH,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    print(f"Evaluated {len(results)} parameter combinations over {oos.index.min().date()} to {oos.index.max().date()}")
    print(f"min_trades_per_month threshold: {MIN_TRADES_PER_MONTH}\n")

    n_pass = results["meets_min_frequency"].sum()
    n_fail = (~results["meets_min_frequency"]).sum()
    print(f"meets_min_frequency: {n_pass} pass, {n_fail} flagged as too illiquid to consider\n")

    cols = [
        "method", "threshold", "hold_days", "overlap_rule",
        "n_trades", "trades_per_month", "meets_min_frequency",
        "net_sharpe", "net_total_return", "gross_sharpe",
    ]

    print("=== TOP 10 BY NET SHARPE, WITHOUT THE FREQUENCY FILTER (illustrates the risk) ===")
    print(results.sort_values("net_sharpe", ascending=False)[cols].head(10).to_string(index=False))
    print()

    print("=== TOP 10 BY NET SHARPE, RESTRICTED TO meets_min_frequency == True ===")
    eligible = results[results["meets_min_frequency"]]
    print(eligible.sort_values("net_sharpe", ascending=False)[cols].head(10).to_string(index=False))
    print()

    print("=== EXAMPLES FLAGGED AS TOO ILLIQUID (meets_min_frequency == False) ===")
    flagged = results[~results["meets_min_frequency"]].sort_values("net_sharpe", ascending=False)
    print(flagged[cols].head(10).to_string(index=False))
