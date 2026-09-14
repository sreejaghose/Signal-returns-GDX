import pandas as pd

from .data import PRICE_COLUMNS

TARGET_COL = "target_gdx_ret_tp1"


def build_features(
    prices: pd.DataFrame,
    max_lag: int,
    lookback: int,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a feature DataFrame from daily close prices.

    All features at row t use only data available through day t (inclusive),
    so they can legitimately be used to predict GDX's t+1 return. The target
    column is GDX's return realized on day t+1, obtained via shift(-1) of
    the t-indexed return series — the last row's target is therefore NaN.

    Args:
        prices: DataFrame of daily closes indexed by Date, one column per ticker.
        max_lag: number of lagged daily-return features to generate per series
            (lags 1..max_lag).
        lookback: rolling window length (in trading days) used for the rolling
            mean/std/z-score and cross-asset rolling-correlation/beta features.
        cols: tickers to build per-series features for. Defaults to PRICE_COLUMNS.

    Returns:
        Feature DataFrame indexed by Date, with a target column named TARGET_COL.
    """
    if cols is None:
        cols = PRICE_COLUMNS
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    returns = prices[cols].pct_change()
    features = {}

    for c in cols:
        ret = returns[c]

        for lag in range(1, max_lag + 1):
            features[f"{c}_ret_lag{lag}"] = ret.shift(lag)

        roll_mean = ret.rolling(lookback).mean()
        roll_std = ret.rolling(lookback).std()
        features[f"{c}_ret_roll_mean_{lookback}"] = roll_mean
        features[f"{c}_ret_roll_std_{lookback}"] = roll_std
        features[f"{c}_ret_zscore_{lookback}"] = (ret - roll_mean) / roll_std

    features["gdx_spy_spread"] = returns["GDX"] - returns["SPY"]
    features["gdx_gld_spread"] = returns["GDX"] - returns["GLD"]
    features["gdxj_gdx_spread"] = returns["GDXJ"] - returns["GDX"]

    gdx_tlt_corr = returns["GDX"].rolling(lookback).corr(returns["TLT"])
    gdx_std = returns["GDX"].rolling(lookback).std()
    tlt_std = returns["TLT"].rolling(lookback).std()
    features[f"gdx_tlt_beta_{lookback}"] = gdx_tlt_corr * gdx_std / tlt_std

    feature_df = pd.DataFrame(features, index=prices.index)
    feature_df[TARGET_COL] = returns["GDX"].shift(-1)

    return feature_df
