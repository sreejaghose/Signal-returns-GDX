import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import CostModel, apply_costs, build_positions, summarize_gross_vs_net, zscore_threshold_signal

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().parent.parent / "results" / "predictions_lightgbm.csv"
ZSCORE_WINDOW = 20
ZSCORE_THRESHOLD = 1.0
HOLD_DAYS = 3
OVERLAP_RULE = "restart_clock"
SPREAD_BPS_SWEEP = [2, 5, 10]

if __name__ == "__main__":
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH
    predictions = pd.read_csv(predictions_path, index_col="date", parse_dates=True)
    oos = predictions.dropna(subset=["y_pred"])
    y_pred, y_true = oos["y_pred"], oos["y_true"]

    signal = zscore_threshold_signal(y_pred, ZSCORE_WINDOW, ZSCORE_THRESHOLD)
    positions, trades = build_positions(signal, hold_days=HOLD_DAYS, overlap_rule=OVERLAP_RULE)

    print(f"Config: zscore thr={ZSCORE_THRESHOLD}, hold_days={HOLD_DAYS}, overlap_rule={OVERLAP_RULE}")
    print(f"Total trade events logged: {len(trades)}")
    by_action = pd.Series([t.action for t in trades]).value_counts()
    print(f"By action: {dict(by_action)}\n")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x: .6f}")

    for spread_bps in SPREAD_BPS_SWEEP:
        cost_model = CostModel(spread_bps=spread_bps)
        result = apply_costs(positions, y_true, trades, cost_model)
        table = summarize_gross_vs_net(result["gross_return"], result["net_return"], result["cost"])

        print(f"=== spread_bps = {spread_bps} ===")
        print(table.to_string())
        print()

    print("=== COMMISSION HOOK DEMO (spread=5bps + $-equivalent commission=1bp/trade) ===")
    cost_model_with_commission = CostModel(spread_bps=5, commission_per_trade=0.0001)
    result = apply_costs(positions, y_true, trades, cost_model_with_commission)
    table = summarize_gross_vs_net(result["gross_return"], result["net_return"], result["cost"])
    print(table.to_string())
