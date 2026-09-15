import numpy as np
import pandas as pd
import pytest

from signal_gdx import CostModel, build_positions, compute_trades_per_month, evaluate_signal_grid

# ~2 years of business days
DATES_2Y = pd.date_range("2020-01-01", periods=504, freq="B")


def test_compute_trades_per_month_counts_only_initiation_actions():
    # one open+close pair, one restart, one flip, one double_down -- only
    # open/restart/flip should count toward n_trades
    signal = pd.Series(0, index=DATES_2Y[:10])
    signal.iloc[0] = 1  # open
    signal.iloc[1] = 1  # restart (same direction, overlapping)
    signal.iloc[2] = -1  # flip
    positions, trades = build_positions(signal, hold_days=5, overlap_rule="restart_clock")

    result = compute_trades_per_month(trades, DATES_2Y[:10])
    assert result["n_trades"] == 3  # open, restart, flip -- not the eventual close
    assert {t.action for t in trades} <= {"open", "restart", "flip", "close"}


def test_compute_trades_per_month_excludes_double_down_and_close():
    signal = pd.Series(0, index=DATES_2Y[:10])
    signal.iloc[0] = 1  # open
    signal.iloc[1] = 1  # double_down (same direction)
    positions, trades = build_positions(signal, hold_days=5, overlap_rule="double_down")

    result = compute_trades_per_month(trades, DATES_2Y[:10])
    assert result["n_trades"] == 1  # only the open; double_down and close don't count


def test_frequency_rate_matches_known_calendar_span():
    # exactly one trade every ~2 months over a 2-year span -> ~0.5 trades/month
    signal = pd.Series(0, index=DATES_2Y)
    trade_dates = DATES_2Y[::42]  # roughly every 2 months of business days
    signal.loc[trade_dates] = 1
    positions, trades = build_positions(signal, hold_days=1, overlap_rule="ignore_new_signal")

    result = compute_trades_per_month(trades, DATES_2Y)
    span_months = (DATES_2Y[-1] - DATES_2Y[0]).days / (365.25 / 12)
    expected_rate = len(trade_dates) / span_months
    assert np.isclose(result["trades_per_month"], expected_rate, rtol=0.05)


def test_meets_min_frequency_flag_respects_custom_threshold():
    signal = pd.Series(0, index=DATES_2Y)
    signal.iloc[[0, 100, 200, 300]] = 1  # 4 trades over ~2 years -> ~0.17/month
    positions, trades = build_positions(signal, hold_days=1, overlap_rule="ignore_new_signal")

    default_result = compute_trades_per_month(trades, DATES_2Y)  # default threshold 1.0
    assert default_result["meets_min_frequency"] is False

    lenient_result = compute_trades_per_month(trades, DATES_2Y, min_trades_per_month=0.1)
    assert lenient_result["meets_min_frequency"] is True


def test_zero_trades_is_a_real_zero_not_nan_and_fails_default_threshold():
    signal = pd.Series(0, index=DATES_2Y)
    positions, trades = build_positions(signal, hold_days=1, overlap_rule="ignore_new_signal")

    result = compute_trades_per_month(trades, DATES_2Y)
    assert result["n_trades"] == 0
    assert result["trades_per_month"] == 0.0
    assert result["meets_min_frequency"] is False


def test_evaluate_signal_grid_reports_frequency_for_every_row():
    rng = np.random.default_rng(0)
    y_pred = pd.Series(rng.normal(0, 0.002, size=len(DATES_2Y)), index=DATES_2Y)
    y_true = pd.Series(rng.normal(0, 0.01, size=len(DATES_2Y)), index=DATES_2Y)

    results = evaluate_signal_grid(
        y_pred,
        y_true,
        raw_thresholds=[0.001, 0.005],
        zscore_thresholds=[1.0],
        zscore_window=20,
        hold_days_grid=[1, 5],
        cost_model=CostModel(spread_bps=5),
        overlap_rules=["ignore_new_signal", "restart_clock"],
    )

    n_signal_configs = 2 + 1  # 2 raw thresholds + 1 zscore threshold
    expected_rows = n_signal_configs * 2 * 2  # x hold_days_grid x overlap_rules
    assert len(results) == expected_rows
    assert {"trades_per_month", "meets_min_frequency", "n_trades", "total_cost_paid"} <= set(results.columns)
    assert results["trades_per_month"].notna().all()
    assert results["meets_min_frequency"].isin([True, False]).all()


def test_evaluate_signal_grid_flags_sparse_high_threshold_combo_as_impractical():
    rng = np.random.default_rng(1)
    y_pred = pd.Series(rng.normal(0, 0.001, size=len(DATES_2Y)), index=DATES_2Y)
    y_true = pd.Series(rng.normal(0, 0.01, size=len(DATES_2Y)), index=DATES_2Y)

    # an absurdly high raw threshold should fire rarely (if ever) over 2 years
    results = evaluate_signal_grid(
        y_pred,
        y_true,
        raw_thresholds=[0.0, 10.0],  # 0.0 fires on almost every day; 10.0 essentially never
        zscore_thresholds=[],
        zscore_window=20,
        hold_days_grid=[1],
        cost_model=CostModel(spread_bps=5),
        overlap_rules=["ignore_new_signal"],
    )

    frequent_row = results[results["threshold"] == 0.0].iloc[0]
    sparse_row = results[results["threshold"] == 10.0].iloc[0]

    assert bool(frequent_row["meets_min_frequency"]) is True
    assert sparse_row["n_trades"] == 0
    assert bool(sparse_row["meets_min_frequency"]) is False


def test_evaluate_signal_grid_min_trades_per_month_is_configurable():
    rng = np.random.default_rng(2)
    y_pred = pd.Series(rng.normal(0, 0.002, size=len(DATES_2Y)), index=DATES_2Y)
    y_true = pd.Series(rng.normal(0, 0.01, size=len(DATES_2Y)), index=DATES_2Y)

    lenient = evaluate_signal_grid(
        y_pred,
        y_true,
        raw_thresholds=[0.004],
        zscore_thresholds=[],
        zscore_window=20,
        hold_days_grid=[1],
        cost_model=CostModel(spread_bps=5),
        overlap_rules=["ignore_new_signal"],
        min_trades_per_month=0.01,
    )
    strict = evaluate_signal_grid(
        y_pred,
        y_true,
        raw_thresholds=[0.004],
        zscore_thresholds=[],
        zscore_window=20,
        hold_days_grid=[1],
        cost_model=CostModel(spread_bps=5),
        overlap_rules=["ignore_new_signal"],
        min_trades_per_month=100.0,
    )
    # same underlying trade count, different verdicts purely from the threshold parameter
    assert lenient.iloc[0]["n_trades"] == strict.iloc[0]["n_trades"]
    assert bool(lenient.iloc[0]["meets_min_frequency"]) is True
    assert bool(strict.iloc[0]["meets_min_frequency"]) is False
