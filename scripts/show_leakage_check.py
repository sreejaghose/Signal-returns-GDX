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
    returns = prices.pct_change()

    sample_positions = [30, 500, 1500, 2500, len(prices) - 2]

    pd.set_option("display.width", 160)

    print("=== ROLLING WINDOW ALIGNMENT (GDX_ret_roll_mean_{}) ===".format(LOOKBACK))
    print("For each sample row t: the L-day window used, and the date it ends on.\n")
    for pos in sample_positions:
        date_t = prices.index[pos]
        date_tp1 = prices.index[pos + 1]
        window_dates = returns.index[pos - LOOKBACK + 1 : pos + 1]
        recomputed_mean = returns["GDX"].iloc[pos - LOOKBACK + 1 : pos + 1].mean()
        stored_mean = feature_df.loc[date_t, f"GDX_ret_roll_mean_{LOOKBACK}"]
        print(f"row t = {date_t.date()}  (next row t+1 = {date_tp1.date()})")
        print(f"  window: {window_dates[0].date()} .. {window_dates[-1].date()}  ({len(window_dates)} days, last day == t: {window_dates[-1] == date_t})")
        print(f"  recomputed mean = {recomputed_mean:.6f}   stored feature = {stored_mean:.6f}   match: {abs(recomputed_mean - stored_mean) < 1e-12}")
        print()

    print("=== LAG-1 FEATURE ALIGNMENT (GDX_ret_lag1) ===\n")
    for pos in sample_positions:
        date_t = prices.index[pos]
        date_t_minus_1 = prices.index[pos - 1]
        stored_lag1 = feature_df.loc[date_t, "GDX_ret_lag1"]
        recomputed = returns["GDX"].iloc[pos - 1]
        print(f"row t = {date_t.date()}  ->  GDX_ret_lag1 sources return realized on {date_t_minus_1.date()}: "
              f"stored={stored_lag1:.6f}  recomputed={recomputed:.6f}  match: {abs(stored_lag1 - recomputed) < 1e-12}")
    print()

    print("=== TARGET ALIGNMENT (target_gdx_ret_tp1) ===\n")
    print(f"{'date_t':<12} {'GDX_px_t':>10} {'date_t+1':<12} {'GDX_px_t+1':>11} {'actual ret t+1':>15} {'stored target':>15} {'match':>6}")
    for pos in sample_positions:
        date_t = prices.index[pos]
        date_tp1 = prices.index[pos + 1]
        px_t = prices["GDX"].iloc[pos]
        px_tp1 = prices["GDX"].iloc[pos + 1]
        actual_ret_tp1 = px_tp1 / px_t - 1
        stored_target = feature_df.loc[date_t, TARGET_COL]
        match = abs(actual_ret_tp1 - stored_target) < 1e-12
        print(f"{date_t.date()!s:<12} {px_t:>10.4f} {date_tp1.date()!s:<12} {px_tp1:>11.4f} {actual_ret_tp1:>15.6f} {stored_target:>15.6f} {match!s:>6}")

    print()
    last_date = prices.index[-1]
    print(f"Last row ({last_date.date()}) target is NaN (no t+1 exists): "
          f"{pd.isna(feature_df.loc[last_date, TARGET_COL])}")
