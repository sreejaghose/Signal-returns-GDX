import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import (
    TARGET_COL,
    build_features,
    load_prices,
    make_elasticnet_model,
    make_shallow_lightgbm,
    walk_forward_backtest,
)

DEFAULT_DATA_PATH = os.environ.get(
    "GDX_DATA_PATH",
    "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx",
)
MAX_LAG = 5
LOOKBACK = 20


@pytest.fixture(scope="module")
def feature_df():
    prices = load_prices(DEFAULT_DATA_PATH)
    return build_features(prices, max_lag=MAX_LAG, lookback=LOOKBACK)


@pytest.fixture(scope="module")
def feature_cols(feature_df):
    return [c for c in feature_df.columns if c != TARGET_COL]


@pytest.fixture(scope="module")
def small_xy(feature_df, feature_cols):
    valid = feature_df.dropna(subset=feature_cols + [TARGET_COL])
    X = valid[feature_cols].to_numpy()[:400]
    y = valid[TARGET_COL].to_numpy()[:400]
    return X, y


def test_elasticnet_model_is_pipeline_wrapping_elasticnetcv():
    model = make_elasticnet_model()
    assert isinstance(model, Pipeline)
    inner = model.named_steps["model"]
    assert isinstance(inner, ElasticNetCV)
    assert isinstance(inner.cv, TimeSeriesSplit)


def test_elasticnet_sweeps_multiple_l1_ratios_and_alphas(small_xy):
    X, y = small_xy
    model = make_elasticnet_model(l1_ratios=[0.2, 0.5, 0.8], n_alphas=10, inner_cv_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    inner = model.named_steps["model"]
    assert inner.l1_ratio == [0.2, 0.5, 0.8]
    # ElasticNetCV records the selected alpha/l1_ratio after fitting, and the
    # full per-(l1_ratio, alpha) MSE grid it searched over
    assert inner.alpha_ > 0
    assert inner.l1_ratio_ in [0.2, 0.5, 0.8]
    assert np.asarray(inner.mse_path_).shape[0] == 3


def test_elasticnet_clones_and_fits_predicts(small_xy):
    X, y = small_xy
    model = clone(make_elasticnet_model(n_alphas=10, inner_cv_splits=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_shallow_lightgbm_is_actually_shallow_and_regularized():
    model = make_shallow_lightgbm()
    params = model.get_params()
    assert params["max_depth"] <= 4
    assert params["num_leaves"] <= 15
    assert params["reg_alpha"] > 0
    assert params["reg_lambda"] > 0
    assert params["n_estimators"] <= 100


def test_lightgbm_clones_and_fits_predicts(small_xy):
    X, y = small_xy
    model = clone(make_shallow_lightgbm())
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_lightgbm_overrides_are_applied():
    model = make_shallow_lightgbm(n_estimators=10)
    assert model.get_params()["n_estimators"] == 10


@pytest.mark.parametrize("model_name", ["elasticnet", "lightgbm"])
def test_each_model_runs_through_the_full_backtest_harness(feature_df, feature_cols, model_name):
    """Both wired-up models are genuine drop-ins for the model-agnostic
    walk-forward harness: fit, predict, and store predicted vs. actual
    next-day GDX return for every fold, with no code changes to the harness."""
    estimator = (
        make_elasticnet_model(n_alphas=5, inner_cv_splits=3)
        if model_name == "elasticnet"
        else make_shallow_lightgbm(n_estimators=10)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions, folds = walk_forward_backtest(
            feature_df,
            feature_cols,
            TARGET_COL,
            estimator,
            min_train_size=700,
            refit_freq=250,
            embargo=LOOKBACK,
            window="expanding",
        )

    assert len(folds) >= 2
    oos = predictions.dropna(subset=["y_pred"])
    assert len(oos) > 0
    assert oos["y_true"].notna().all()
    assert np.all(np.isfinite(oos["y_pred"]))
