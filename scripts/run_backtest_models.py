import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import (
    TARGET_COL,
    build_features,
    load_prices,
    make_elasticnet_model,
    make_shallow_lightgbm,
    walk_forward_backtest,
)

DEFAULT_DATA_PATH = "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx"
MAX_LAG = 5
LOOKBACK = 20
MIN_TRAIN_SIZE = 750
REFIT_FREQ = 21
EMBARGO = LOOKBACK
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"


def summarize(name: str, predictions: pd.DataFrame, folds) -> None:
    oos = predictions.dropna(subset=["y_pred"])
    err = oos["y_pred"] - oos["y_true"]
    hit_rate = (np.sign(oos["y_pred"]) == np.sign(oos["y_true"])).mean()

    print(f"=== {name} ===")
    print(f"Folds: {len(folds)}   OOS rows: {len(oos)}   Range: {oos.index.min().date()} to {oos.index.max().date()}")
    print(f"MAE:  {err.abs().mean():.6f}")
    print(f"RMSE: {np.sqrt((err ** 2).mean()):.6f}")
    print(f"Directional hit rate: {hit_rate:.4f}")
    print()
    print(oos.head(3).to_string())
    print("...")
    print(oos.tail(3).to_string())
    print()


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    prices = load_prices(data_path)
    feature_df = build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)
    feature_cols = [c for c in feature_df.columns if c != TARGET_COL]

    OUTPUT_DIR.mkdir(exist_ok=True)

    models = {
        "elasticnet": make_elasticnet_model(),
        "lightgbm": make_shallow_lightgbm(),
    }

    for name, estimator in models.items():
        predictions, folds = walk_forward_backtest(
            feature_df,
            feature_cols,
            TARGET_COL,
            estimator,
            min_train_size=MIN_TRAIN_SIZE,
            refit_freq=REFIT_FREQ,
            embargo=EMBARGO,
            window="expanding",
        )
        summarize(name, predictions, folds)

        out_path = OUTPUT_DIR / f"predictions_{name}.csv"
        predictions.to_csv(out_path, index_label="date")
        print(f"Saved predicted vs. actual to {out_path}\n")
