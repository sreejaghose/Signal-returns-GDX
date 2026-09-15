import numpy as np
import pandas as pd
import pytest

from signal_gdx import (
    build_positions,
    compute_prediction_zscore,
    raw_threshold_signal,
    sweep_signal_thresholds,
    zscore_threshold_signal,
)

DATES7 = pd.date_range("2020-01-01", periods=7, freq="B")


def test_raw_threshold_signal_basic_values():
    y_pred = pd.Series([0.5, -0.5, 0.05, -0.05, 0.0, np.nan], index=pd.date_range("2020-01-01", periods=6, freq="B"))
    signal = raw_threshold_signal(y_pred, threshold=0.1)
    assert signal.tolist() == [1, -1, 0, 0, 0, 0]


def test_raw_threshold_rejects_negative_threshold():
    y_pred = pd.Series([0.1], index=pd.date_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        raw_threshold_signal(y_pred, threshold=-0.1)


def test_compute_prediction_zscore_matches_manual_calculation():
    y_pred = pd.Series(np.linspace(-1, 1, 30), index=pd.date_range("2020-01-01", periods=30, freq="B"))
    window = 5
    z = compute_prediction_zscore(y_pred, window)

    assert z.iloc[: window - 1].isna().all()

    pos = 15
    manual_window = y_pred.iloc[pos - window + 1 : pos + 1]
    expected_z = (y_pred.iloc[pos] - manual_window.mean()) / manual_window.std()
    assert np.isclose(z.iloc[pos], expected_z)


def test_compute_prediction_zscore_is_causal():
    """Truncating the series after t must not change z_t (no lookahead)."""
    rng = np.random.default_rng(0)
    y_pred = pd.Series(rng.normal(size=100), index=pd.date_range("2020-01-01", periods=100, freq="B"))
    window = 10
    full_z = compute_prediction_zscore(y_pred, window)
    truncated_z = compute_prediction_zscore(y_pred.iloc[:60], window)
    pd.testing.assert_series_equal(full_z.iloc[:60], truncated_z, check_names=False)


def test_zscore_threshold_signal_thresholds_correctly():
    y_pred = pd.Series(np.linspace(-1, 1, 30), index=pd.date_range("2020-01-01", periods=30, freq="B"))
    window = 5
    z = compute_prediction_zscore(y_pred, window)
    signal = zscore_threshold_signal(y_pred, window, threshold=1.0)

    assert (signal[z > 1.0] == 1).all()
    assert (signal[z < -1.0] == -1).all()
    assert (signal[z.abs() <= 1.0] == 0).all()
    assert (signal[z.isna()] == 0).all()


def test_sweep_signal_thresholds_matches_individual_calls():
    y_pred = pd.Series(np.linspace(-1, 1, 30), index=pd.date_range("2020-01-01", periods=30, freq="B"))
    raw_thresholds = [0.1, 0.3]
    zscore_thresholds = [0.5, 1.5]
    window = 5

    grid = sweep_signal_thresholds(y_pred, raw_thresholds, zscore_thresholds, window)

    assert set(grid.keys()) == {("raw", 0.1), ("raw", 0.3), ("zscore", 0.5), ("zscore", 1.5)}
    pd.testing.assert_series_equal(grid[("raw", 0.1)], raw_threshold_signal(y_pred, 0.1))
    pd.testing.assert_series_equal(grid[("zscore", 1.5)], zscore_threshold_signal(y_pred, window, 1.5))


@pytest.mark.parametrize("overlap_rule", ["ignore_new_signal", "restart_clock", "double_down"])
def test_hold_days_one_collapses_to_raw_signal(overlap_rule):
    signal = pd.Series([1, -1, 0, 1, 1, -1, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=1, overlap_rule=overlap_rule)
    assert positions.tolist() == signal.tolist()


def test_no_signal_produces_no_trades():
    signal = pd.Series([0] * 7, index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")
    assert positions.tolist() == [0] * 7
    assert trades == []


def test_ignore_new_signal_ignores_any_overlap_regardless_of_direction():
    signal = pd.Series([1, -1, 1, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")
    # opening +1 on day0 rides out untouched through day2, then flat
    assert positions.tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert [t.action for t in trades] == ["open", "close"]
    assert trades[0].direction == 1 and trades[0].size == 1


def test_restart_clock_same_direction_extends_hold():
    signal = pd.Series([1, 1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="restart_clock")
    # day1's same-direction signal restarts the clock -> holds through day3
    assert positions.tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert [t.action for t in trades] == ["open", "restart", "close"]


def test_restart_clock_opposite_direction_flips_and_resets_clock():
    signal = pd.Series([1, -1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="restart_clock")
    assert positions.tolist() == [1, -1, -1, -1, 0, 0, 0]
    assert [t.action for t in trades] == ["open", "flip", "close"]
    assert trades[1].direction == -1 and trades[1].size == 1


def test_double_down_same_direction_stacks_size_without_resetting_clock():
    signal = pd.Series([1, 1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="double_down")
    # size doubles on day1 but the original 3-day clock (started day0) is untouched
    assert positions.tolist() == [1, 2, 2, 0, 0, 0, 0]
    assert [t.action for t in trades] == ["open", "double_down", "close"]
    assert trades[1].size == 2


def test_double_down_does_not_stack_past_2x():
    signal = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0], index=pd.date_range("2020-01-01", periods=9, freq="B"))
    positions, trades = build_positions(signal, hold_days=5, overlap_rule="double_down")
    assert positions.max() == 2
    double_down_trades = [t for t in trades if t.action == "double_down"]
    assert all(t.size == 2 for t in double_down_trades)


def test_double_down_opposite_direction_flips_like_restart():
    signal = pd.Series([1, -1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="double_down")
    assert positions.tolist() == [1, -1, -1, -1, 0, 0, 0]
    assert [t.action for t in trades] == ["open", "flip", "close"]


def test_execution_uses_same_day_as_signal_no_extra_shift():
    signal = pd.Series([0, 1, 0, 0, 0], index=pd.date_range("2020-01-01", periods=5, freq="B"))
    positions, trades = build_positions(signal, hold_days=1, overlap_rule="ignore_new_signal")
    assert positions.iloc[1] == 1
    assert trades[0].date == signal.index[1]


def test_invalid_hold_days_and_overlap_rule_raise():
    signal = pd.Series([1, 0], index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(ValueError):
        build_positions(signal, hold_days=0, overlap_rule="ignore_new_signal")
    with pytest.raises(ValueError):
        build_positions(signal, hold_days=1, overlap_rule="not_a_real_rule")
