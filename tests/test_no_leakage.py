import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import PRICE_COLUMNS, TARGET_COL, build_features, load_prices

DEFAULT_DATA_PATH = os.environ.get(
    "GDX_DATA_PATH",
    "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx",
)
MAX_LAG = 5
LOOKBACK = 20
N_SAMPLES = 15


@pytest.fixture(scope="module")
def prices():
    return load_prices(DEFAULT_DATA_PATH)


@pytest.fixture(scope="module")
def feature_df(prices):
    return build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)


def test_truncation_invariance_of_features(prices, feature_df):
    """Gold-standard no-lookahead check.

    If row t's features truly depend only on data through day t, then
    deleting every row after t and recomputing must leave row t unchanged.
    If any feature (lags, rolling stats, spreads, beta) secretly reached
    into t+1 or later, this recomputation would differ.
    """
    rng = np.random.default_rng(0)
    n = len(prices)
    candidate_positions = range(LOOKBACK + MAX_LAG, n - 1)
    sample_positions = rng.choice(list(candidate_positions), size=N_SAMPLES, replace=False)

    feature_cols = [c for c in feature_df.columns if c != TARGET_COL]

    for pos in sample_positions:
        truncated_prices = prices.iloc[: pos + 1]
        truncated_features = build_features(truncated_prices, max_lag=MAX_LAG, lookback=LOOKBACK)

        full_row = feature_df.iloc[pos][feature_cols]
        trunc_row = truncated_features.iloc[pos][feature_cols]

        pd.testing.assert_series_equal(full_row, trunc_row, check_names=False, check_exact=False)


def test_target_equals_next_day_return(prices, feature_df):
    """Target at row t must equal GDX's realized return on day t+1, exactly,
    and must require nothing beyond t+1 to reproduce."""
    rng = np.random.default_rng(1)
    n = len(prices)
    sample_positions = rng.choice(range(0, n - 1), size=N_SAMPLES, replace=False)

    for pos in sample_positions:
        date_t = prices.index[pos]

        actual_next_return = prices["GDX"].iloc[pos + 1] / prices["GDX"].iloc[pos] - 1
        target_value = feature_df.loc[date_t, TARGET_COL]
        assert np.isclose(target_value, actual_next_return)

        # Truncating right at t+1 (the minimum info needed) must reproduce
        # the same target -- confirms nothing beyond t+1 is required.
        truncated_prices = prices.iloc[: pos + 2]
        truncated_features = build_features(truncated_prices, max_lag=MAX_LAG, lookback=LOOKBACK)
        assert np.isclose(truncated_features.loc[date_t, TARGET_COL], target_value)


def test_last_row_target_is_nan(feature_df):
    """No t+1 exists for the final date, so its target must be NaN, not
    silently filled or wrapped around."""
    assert pd.isna(feature_df[TARGET_COL].iloc[-1])


def test_rolling_and_beta_window_boundaries(prices, feature_df):
    """Explicitly checks the rolling mean/std/z-score and GDX-TLT beta use a
    window of the L most recent returns ending at t (inclusive), never t+1."""
    returns = prices[PRICE_COLUMNS].pct_change()
    rng = np.random.default_rng(2)
    n = len(prices)
    sample_positions = rng.choice(range(LOOKBACK, n - 1), size=10, replace=False)

    for pos in sample_positions:
        date_t = prices.index[pos]

        window = returns["GDX"].iloc[pos - LOOKBACK + 1 : pos + 1]
        assert len(window) == LOOKBACK
        assert window.index[-1] == date_t
        assert window.index[-1] < prices.index[pos + 1]

        expected_mean = window.mean()
        expected_std = window.std()
        expected_z = (returns["GDX"].iloc[pos] - expected_mean) / expected_std

        assert np.isclose(feature_df.loc[date_t, f"GDX_ret_roll_mean_{LOOKBACK}"], expected_mean)
        assert np.isclose(feature_df.loc[date_t, f"GDX_ret_roll_std_{LOOKBACK}"], expected_std)
        assert np.isclose(feature_df.loc[date_t, f"GDX_ret_zscore_{LOOKBACK}"], expected_z)

        tlt_window = returns["TLT"].iloc[pos - LOOKBACK + 1 : pos + 1]
        expected_beta = window.corr(tlt_window) * window.std() / tlt_window.std()
        assert np.isclose(feature_df.loc[date_t, f"gdx_tlt_beta_{LOOKBACK}"], expected_beta)


def test_lag_features_point_strictly_backward(prices, feature_df):
    """Lag-k feature at row t must equal the return realized k days earlier,
    never anything at or after t."""
    returns = prices[PRICE_COLUMNS].pct_change()
    rng = np.random.default_rng(3)
    n = len(prices)
    sample_positions = rng.choice(range(MAX_LAG, n), size=10, replace=False)

    for pos in sample_positions:
        date_t = prices.index[pos]
        for lag in range(1, MAX_LAG + 1):
            expected = returns["GDX"].iloc[pos - lag]
            actual = feature_df.loc[date_t, f"GDX_ret_lag{lag}"]
            if pd.isna(expected):
                assert pd.isna(actual)
            else:
                assert np.isclose(actual, expected)
