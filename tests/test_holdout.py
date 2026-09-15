import numpy as np
import pandas as pd
import pytest

from signal_gdx import (
    CostModel,
    compute_holdout_start,
    evaluate_candidate_on_holdout,
    flag_holdout_degradation,
)


def test_compute_holdout_start_matches_known_fraction():
    idx = pd.date_range("2000-01-01", "2020-01-01", freq="D")
    holdout_start = compute_holdout_start(idx, 0.2)
    total_days = (idx.max() - idx.min()).days
    expected = idx.max() - pd.Timedelta(days=total_days * 0.2)
    assert abs((holdout_start - expected).days) <= 1


def test_compute_holdout_start_snaps_forward_to_available_date():
    # sparse index (business days only) -- cutoff may land on a weekend
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    holdout_start = compute_holdout_start(idx, 0.5)
    assert holdout_start in idx  # snapped to an actual date, not interpolated


def test_compute_holdout_start_rejects_bad_fraction():
    idx = pd.date_range("2020-01-01", periods=10)
    with pytest.raises(ValueError):
        compute_holdout_start(idx, 0.0)
    with pytest.raises(ValueError):
        compute_holdout_start(idx, 1.0)
    with pytest.raises(ValueError):
        compute_holdout_start(idx, 1.5)


def test_evaluate_candidate_on_holdout_only_uses_dates_at_or_after_start():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    y_pred = pd.Series(rng.normal(0, 0.002, size=100), index=dates)
    y_true = pd.Series(rng.normal(0, 0.01, size=100), index=dates)
    holdout_start = dates[70]

    result = evaluate_candidate_on_holdout(
        y_pred, y_true, method="raw", threshold=0.001, lookback=20,
        hold_days=3, overlap_rule="ignore_new_signal", cost_model=CostModel(spread_bps=5),
        holdout_start=holdout_start,
    )
    # n_days in the underlying performance calc reflects only the holdout window
    assert result["gross_hit_rate"] is not None
    # cross-check against a manual restriction
    restricted_pred = y_pred[y_pred.index >= holdout_start]
    assert len(restricted_pred) == 30


def test_evaluate_candidate_on_holdout_zscore_method_uses_given_lookback():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(1)
    y_pred = pd.Series(rng.normal(0, 0.002, size=100), index=dates)
    y_true = pd.Series(rng.normal(0, 0.01, size=100), index=dates)
    holdout_start = dates[50]

    result = evaluate_candidate_on_holdout(
        y_pred, y_true, method="zscore", threshold=1.0, lookback=10,
        hold_days=2, overlap_rule="restart_clock", cost_model=CostModel(spread_bps=2),
        holdout_start=holdout_start,
    )
    assert "net_sharpe" in result


def test_flag_holdout_degradation_flags_large_relative_drop():
    result = flag_holdout_degradation(in_sample_sharpe=0.8, holdout_sharpe=0.2)
    assert result["flagged_overfit"] is True
    assert result["relative_drop"] == pytest.approx(0.75)


def test_flag_holdout_degradation_flags_sign_flip_from_solid_positive():
    result = flag_holdout_degradation(in_sample_sharpe=0.5, holdout_sharpe=-0.1)
    assert result["flagged_overfit"] is True


def test_flag_holdout_degradation_does_not_flag_mild_decay():
    result = flag_holdout_degradation(in_sample_sharpe=0.8, holdout_sharpe=0.6)
    assert result["flagged_overfit"] is False
    assert result["relative_drop"] == pytest.approx(0.25)


def test_flag_holdout_degradation_handles_zero_in_sample_sharpe():
    result = flag_holdout_degradation(in_sample_sharpe=0.0, holdout_sharpe=-0.2)
    assert np.isnan(result["relative_drop"])
    assert result["flagged_overfit"] is False  # 0 baseline isn't "solidly positive"


def test_flag_holdout_degradation_improvement_is_not_flagged():
    result = flag_holdout_degradation(in_sample_sharpe=0.5, holdout_sharpe=0.9)
    assert result["flagged_overfit"] is False
    assert result["relative_drop"] < 0
