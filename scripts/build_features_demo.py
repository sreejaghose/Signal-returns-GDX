import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import TARGET_COL, build_features, load_prices

DEFAULT_DATA_PATH = "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx"
MAX_LAG = 5
LOOKBACK = 20

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    prices = load_prices(data_path)
    feature_df = build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)

    feature_cols = [c for c in feature_df.columns if c != TARGET_COL]
    print(f"=== FEATURE LIST ({len(feature_cols)} features + 1 target) ===")
    for c in feature_cols:
        print(f"  {c}")
    print(f"  {TARGET_COL}  (target)")
    print()

    print(f"=== SHAPE: {feature_df.shape} ===")
    print()

    pd.set_option("display.max_rows", 200)
    null_counts = feature_df.isna().sum()
    print("=== NULL COUNT PER COLUMN (after shifting) ===")
    print(null_counts.to_string())
    print()
    print(f"Total rows: {len(feature_df)}")
    print(f"Rows with zero NaNs across all columns: {(~feature_df.isna().any(axis=1)).sum()}")
