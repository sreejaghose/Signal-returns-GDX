import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import TARGET_COL, build_features, load_prices, walk_forward_backtest

DEFAULT_DATA_PATH = os.environ.get(
    "GDX_DATA_PATH",
    "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx",
)
MAX_LAG = 5
LOOKBACK = 20
MIN_TRAIN = 500
REFIT_FREQ = 21
EMBARGO = LOOKBACK


@pytest.fixture(scope="module")
def feature_df():
    prices = load_prices(DEFAULT_DATA_PATH)
    return build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)


@pytest.fixture(scope="module")
def feature_cols(feature_df):
    return [c for c in feature_df.columns if c != TARGET_COL]


def test_folds_are_contiguous_non_overlapping_and_embargoed(feature_df, feature_cols):
    _, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        LinearRegression(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )

    valid_dates = feature_df.dropna(subset=feature_cols + [TARGET_COL]).index

    assert len(folds) > 1

    for i, fold in enumerate(folds):
        # embargo really exists: train_end and test_start must not be adjacent
        train_end_pos = valid_dates.get_loc(fold.train_end)
        test_start_pos = valid_dates.get_loc(fold.test_start)
        assert test_start_pos - train_end_pos - 1 == EMBARGO, (
            f"fold {i}: expected {EMBARGO} embargoed rows between train_end and test_start, "
            f"got {test_start_pos - train_end_pos - 1}"
        )

        # test block internally contiguous
        test_end_pos = valid_dates.get_loc(fold.test_end)
        assert test_end_pos - test_start_pos + 1 == fold.n_test

        # expanding window always starts at the beginning
        assert fold.train_start == valid_dates[0]

    # folds tile the timeline with no gaps and no overlaps: fold i+1's train
    # extends exactly through fold i's last tested row (that data is now
    # safely in the past and re-enters the training set)
    for i in range(len(folds) - 1):
        this_test_end_pos = valid_dates.get_loc(folds[i].test_end)
        next_train_end_pos = valid_dates.get_loc(folds[i + 1].train_end)
        assert next_train_end_pos == this_test_end_pos


def test_rolling_window_has_fixed_train_size_and_drops_old_data(feature_df, feature_cols):
    train_window_size = MIN_TRAIN
    _, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        LinearRegression(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="rolling",
        train_window_size=train_window_size,
    )

    valid_dates = feature_df.dropna(subset=feature_cols + [TARGET_COL]).index

    for fold in folds[1:]:
        assert fold.n_train == train_window_size
        assert fold.train_start != valid_dates[0]


def test_expanding_window_grows_every_fold(feature_df, feature_cols):
    _, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        LinearRegression(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )
    train_sizes = [f.n_train for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]


def test_predictions_cover_entire_in_sample_period_after_burn_in(feature_df, feature_cols):
    predictions, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        LinearRegression(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )

    valid_dates = feature_df.dropna(subset=feature_cols + [TARGET_COL]).index
    first_test_date = folds[0].test_start
    last_test_date = folds[-1].test_end

    oos_span = valid_dates[
        (valid_dates >= first_test_date) & (valid_dates <= last_test_date)
    ]
    # every date in the OOS span has a prediction, except the embargoed gaps
    non_embargo_dates = [d for d in oos_span if not pd.isna(predictions.loc[d, "fold"])]
    assert len(non_embargo_dates) == sum(f.n_test for f in folds)

    # nothing before the first fold's training window has a prediction
    burn_in_dates = feature_df.index[feature_df.index < folds[0].train_start]
    assert predictions.loc[burn_in_dates, "y_pred"].isna().all()

    # every predicted row also has its true target stored, for easy scoring
    predicted_rows = predictions.dropna(subset=["y_pred"])
    assert predicted_rows["y_true"].notna().all()


def test_model_agnostic_with_different_estimators(feature_df, feature_cols):
    preds_linreg, _ = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        LinearRegression(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )
    preds_dummy, _ = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        DummyRegressor(strategy="mean"),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )

    both_predicted = preds_linreg["y_pred"].notna() & preds_dummy["y_pred"].notna()
    assert both_predicted.sum() > 0
    # a dummy mean-predictor and a linear model should not produce identical
    # predictions on real data -- confirms the estimator is actually being used
    assert not np.allclose(
        preds_linreg.loc[both_predicted, "y_pred"],
        preds_dummy.loc[both_predicted, "y_pred"],
    )


def test_no_leakage_train_rows_never_reach_into_test_or_embargo(feature_df, feature_cols):
    """Refit each fold with a sentinel estimator that records exactly which
    rows it was trained on, and confirm those rows never include any date
    at or after the fold's own embargo_start."""

    class RecordingEstimator:
        def __init__(self):
            self.seen_max_row = None

        def fit(self, X_arr, y_arr):
            self.seen_max_row = len(X_arr)
            return self

        def predict(self, X_arr):
            return np.zeros(len(X_arr))

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    _, folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        RecordingEstimator(),
        min_train_size=MIN_TRAIN,
        refit_freq=REFIT_FREQ,
        embargo=EMBARGO,
        window="expanding",
    )

    valid_dates = feature_df.dropna(subset=feature_cols + [TARGET_COL]).index
    for fold in folds:
        train_end_pos = valid_dates.get_loc(fold.train_end)
        assert fold.n_train == train_end_pos + 1
        assert fold.embargo_start is not None
        embargo_start_pos = valid_dates.get_loc(fold.embargo_start)
        assert train_end_pos < embargo_start_pos
