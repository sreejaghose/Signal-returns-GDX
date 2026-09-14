import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import TARGET_COL, build_features, load_prices, walk_forward_backtest

DEFAULT_DATA_PATH = "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx"
MAX_LAG = 5
LOOKBACK = 20
MIN_TRAIN_SIZE = 750
REFIT_FREQ = 21
EMBARGO = LOOKBACK

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    prices = load_prices(data_path)
    feature_df = build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)
    feature_cols = [c for c in feature_df.columns if c != TARGET_COL]

    predictions, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        Ridge(alpha=1.0),
        min_train_size=MIN_TRAIN_SIZE,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )

    print(f"=== {len(folds)} FOLDS (expanding window, refit every {REFIT_FREQ} days, embargo={EMBARGO}) ===\n")
    fold_rows = [
        {
            "fold": f.fold_id,
            "train_start": f.train_start.date(),
            "train_end": f.train_end.date(),
            "n_train": f.n_train,
            "embargo_start": f.embargo_start.date() if f.embargo_start else None,
            "embargo_end": f.embargo_end.date() if f.embargo_end else None,
            "test_start": f.test_start.date(),
            "test_end": f.test_end.date(),
            "n_test": f.n_test,
        }
        for f in folds
    ]
    fold_table = pd.DataFrame(fold_rows)
    pd.set_option("display.width", 160)
    print(fold_table.head(10).to_string(index=False))
    print("...")
    print(fold_table.tail(3).to_string(index=False))
    print()

    oos = predictions.dropna(subset=["y_pred"])
    print(f"=== OUT-OF-FOLD PREDICTIONS: {len(oos)} rows, {oos['fold'].nunique()} folds ===")
    print(f"Date range covered: {oos.index.min().date()} to {oos.index.max().date()}")
    print()

    err = oos["y_pred"] - oos["y_true"]
    print("=== OOS ERROR SUMMARY ===")
    print(f"MAE:  {err.abs().mean():.6f}")
    print(f"RMSE: {np.sqrt((err ** 2).mean()):.6f}")
    hit_rate = (np.sign(oos["y_pred"]) == np.sign(oos["y_true"])).mean()
    print(f"Directional hit rate: {hit_rate:.4f}")
    print()

    print("=== SAMPLE ROWS ===")
    print(oos.head(5).to_string())
    print("...")
    print(oos.tail(5).to_string())
