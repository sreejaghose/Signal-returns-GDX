import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import CostModel, evaluate_parameter_combination, zscore_threshold_signal

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().parent.parent / "results" / "predictions_lightgbm.csv"
ZSCORE_WINDOW = 20
ZSCORE_THRESHOLD = 0.5
HOLD_DAYS = 5
OVERLAP_RULE = "ignore_new_signal"
SPREAD_BPS = 5

if __name__ == "__main__":
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH
    predictions = pd.read_csv(predictions_path, index_col="date", parse_dates=True)
    oos = predictions.dropna(subset=["y_pred"])
    y_pred, y_true = oos["y_pred"], oos["y_true"]

    signal = zscore_threshold_signal(y_pred, ZSCORE_WINDOW, ZSCORE_THRESHOLD)

    row = evaluate_parameter_combination(
        y_pred,
        y_true,
        signal,
        hold_days=HOLD_DAYS,
        overlap_rule=OVERLAP_RULE,
        cost_model=CostModel(spread_bps=SPREAD_BPS),
    )

    decay_gross = row.pop("hold_day_decay_gross")
    decay_net = row.pop("hold_day_decay_net")
    decay_counts = row.pop("hold_day_decay_episode_counts")

    print(f"Config: zscore thr={ZSCORE_THRESHOLD}, hold_days={HOLD_DAYS}, overlap_rule={OVERLAP_RULE}, spread_bps={SPREAD_BPS}\n")

    print("=== SINGLE-ROW SUMMARY ===")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key:<28} {value: .6f}")
        else:
            print(f"{key:<28} {value}")
    print()

    print("=== DAY-OF-HOLD DECAY CURVE (average return isolated to that day of the trade) ===")
    decay_table = pd.DataFrame(
        {
            "gross": pd.Series(decay_gross),
            "net": pd.Series(decay_net),
            "n_episodes_reaching_this_day": pd.Series(decay_counts),
        }
    )
    decay_table.index.name = "day"
    print(decay_table.to_string(float_format=lambda x: f"{x: .6f}"))
