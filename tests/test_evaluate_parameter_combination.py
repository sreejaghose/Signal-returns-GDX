import numpy as np
import pandas as pd
import pytest

from signal_gdx import (
    CostModel,
    build_positions,
    compute_hold_day_decay,
    evaluate_parameter_combination,
)

DATES = pd.date_range("2020-01-01", periods=20, freq="B")


def test_trade_episodes_natural_close_via_decay_curve_day_indexing():
    """A single 3-day hold with no overlap: episode should be exactly 3 days,
    so the decay curve has entries for day 1, 2, 3 and nothing beyond."""
    signal = pd.Series(0, index=DATES)
    signal.iloc[0] = 1
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")
    y_true = pd.Series(0.01, index=DATES)
    gross_return = positions * y_true
    net_return = gross_return.copy()  # no costs for this check

    decay_gross, _, _ = compute_hold_day_decay(DATES, trades, gross_return, net_return)
    assert set(decay_gross.keys()) == {1, 2, 3}
    for day in (1, 2, 3):
        assert np.isclose(decay_gross[day], 0.01)


def test_trade_episode_cut_short_by_restart_ends_before_next_start():
    signal = pd.Series(0, index=DATES)
    signal.iloc[0] = 1  # open
    signal.iloc[1] = 1  # restart same direction on day 2 -- cuts first episode to 1 day
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="restart_clock")
    y_true = pd.Series(np.linspace(0.01, 0.05, len(DATES)), index=DATES)
    gross_return = positions * y_true
    net_return = gross_return.copy()

    decay_gross, _, _ = compute_hold_day_decay(DATES, trades, gross_return, net_return)
    # first episode contributes only 1 day-1 sample; the restarted episode
    # contributes its own day 1, 2, 3 -- so day 1 averages across 2 episodes,
    # days 2 and 3 come from only the restarted episode
    assert set(decay_gross.keys()) == {1, 2, 3}
    expected_day1 = np.mean([gross_return.iloc[0], gross_return.iloc[1]])
    assert np.isclose(decay_gross[1], expected_day1)
    assert np.isclose(decay_gross[2], gross_return.iloc[2])
    assert np.isclose(decay_gross[3], gross_return.iloc[3])


def test_trade_episode_runs_off_end_of_data_without_explicit_close():
    signal = pd.Series(0, index=DATES)
    signal.iloc[-1] = 1  # opens on the very last date, hold_days longer than remaining data
    positions, trades = build_positions(signal, hold_days=5, overlap_rule="ignore_new_signal")
    y_true = pd.Series(0.01, index=DATES)
    gross_return = positions * y_true
    net_return = gross_return.copy()

    # no explicit "close" trade should have been logged (ran out of data)
    assert all(t.action != "close" for t in trades)

    decay_gross, _, _ = compute_hold_day_decay(DATES, trades, gross_return, net_return)
    assert set(decay_gross.keys()) == {1}  # only 1 day of data existed after entry


def test_evaluate_parameter_combination_returns_all_required_metrics():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    y_pred = pd.Series(rng.normal(0, 0.002, size=300), index=dates)
    y_true = pd.Series(rng.normal(0, 0.01, size=300), index=dates)
    signal = pd.Series(np.where(y_pred > 0.001, 1, np.where(y_pred < -0.001, -1, 0)), index=dates)

    row = evaluate_parameter_combination(
        y_pred, y_true, signal, hold_days=3, overlap_rule="restart_clock",
        cost_model=CostModel(spread_bps=5),
    )

    required_keys = {
        "gross_sharpe", "net_sharpe",
        "gross_hit_rate", "net_hit_rate",
        "avg_gross_return_per_trade", "avg_net_return_per_trade",
        "gross_max_drawdown", "net_max_drawdown",
        "annual_turnover", "trades_per_month", "meets_min_frequency",
        "hold_day_decay_gross", "hold_day_decay_net",
    }
    assert required_keys <= set(row.keys())
    assert isinstance(row["hold_day_decay_gross"], dict)
    assert isinstance(row["hold_day_decay_net"], dict)
    assert row["net_sharpe"] <= row["gross_sharpe"]  # costs can only hurt, never help
    assert row["net_max_drawdown"] <= row["gross_max_drawdown"]  # net dd at least as deep


def test_avg_return_per_trade_equals_total_inmarket_return_over_n_trades():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    y_pred = pd.Series(rng.normal(0, 0.002, size=200), index=dates)
    y_true = pd.Series(rng.normal(0, 0.01, size=200), index=dates)
    signal = pd.Series(np.where(y_pred > 0.0015, 1, np.where(y_pred < -0.0015, -1, 0)), index=dates)

    row = evaluate_parameter_combination(
        y_pred, y_true, signal, hold_days=2, overlap_rule="double_down",
        cost_model=CostModel(spread_bps=3),
    )
    positions, trades = build_positions(signal, hold_days=2, overlap_rule="double_down")
    gross_return = positions * y_true
    expected_avg = gross_return.sum() / row["n_trades"]
    assert np.isclose(row["avg_gross_return_per_trade"], expected_avg)


def test_zero_trades_combination_has_nan_avg_return_and_empty_decay_curve():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    y_pred = pd.Series(0.0, index=dates)
    y_true = pd.Series(0.01, index=dates)
    signal = pd.Series(0, index=dates)  # never fires

    row = evaluate_parameter_combination(
        y_pred, y_true, signal, hold_days=3, overlap_rule="ignore_new_signal",
        cost_model=CostModel(spread_bps=5),
    )
    assert row["n_trades"] == 0
    assert np.isnan(row["avg_gross_return_per_trade"])
    assert np.isnan(row["avg_net_return_per_trade"])
    assert row["hold_day_decay_gross"] == {}
    assert row["hold_day_decay_net"] == {}
    assert row["meets_min_frequency"] is False


def test_turnover_scales_with_spread_independent_cost_model():
    """annual_turnover should be identical regardless of spread_bps -- it's a
    measure of trading activity, not of cost paid."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(2)
    y_pred = pd.Series(rng.normal(0, 0.002, size=300), index=dates)
    y_true = pd.Series(rng.normal(0, 0.01, size=300), index=dates)
    signal = pd.Series(np.where(y_pred > 0.001, 1, np.where(y_pred < -0.001, -1, 0)), index=dates)

    row_low = evaluate_parameter_combination(
        y_pred, y_true, signal, hold_days=3, overlap_rule="restart_clock", cost_model=CostModel(spread_bps=2)
    )
    row_high = evaluate_parameter_combination(
        y_pred, y_true, signal, hold_days=3, overlap_rule="restart_clock", cost_model=CostModel(spread_bps=10)
    )
    assert np.isclose(row_low["annual_turnover"], row_high["annual_turnover"])
    assert row_low["total_cost_paid"] < row_high["total_cost_paid"]
