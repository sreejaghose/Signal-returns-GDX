import numpy as np
import pandas as pd
import pytest

from signal_gdx import CostModel, apply_costs, build_positions, compute_transaction_costs

DATES7 = pd.date_range("2020-01-01", periods=7, freq="B")


def test_open_and_close_each_charge_full_size_notional():
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")
    # open (day0, size 1) and close (day2, size 1) -> 2 events, 1 unit each
    assert [t.notional for t in trades] == [1, 1]

    cost_model = CostModel(spread_bps=10)
    costs = compute_transaction_costs(trades, positions.index, cost_model)
    assert np.isclose(costs.loc[DATES7[0]], 1 * 10 / 10_000)
    assert np.isclose(costs.loc[DATES7[2]], 1 * 10 / 10_000)
    assert (costs.drop([DATES7[0], DATES7[2]]) == 0).all()


def test_restart_same_direction_is_still_charged_despite_no_net_exposure_change():
    """A restart is itself a transaction even though net position doesn't change."""
    signal = pd.Series([1, 1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="restart_clock")
    restart_trade = next(t for t in trades if t.action == "restart")
    assert restart_trade.prev_size == 1 and restart_trade.size == 1
    assert restart_trade.notional == 2  # closes the old 1x leg, opens a new 1x leg

    cost_model = CostModel(spread_bps=10)
    costs = compute_transaction_costs(trades, positions.index, cost_model)
    assert np.isclose(costs.loc[DATES7[1]], 2 * 10 / 10_000)


def test_flip_charges_both_legs():
    signal = pd.Series([1, -1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="restart_clock")
    flip_trade = next(t for t in trades if t.action == "flip")
    assert flip_trade.prev_direction == 1 and flip_trade.direction == -1
    assert flip_trade.notional == 2


def test_double_down_only_charges_the_incremental_unit():
    signal = pd.Series([1, 1, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="double_down")
    dd_trade = next(t for t in trades if t.action == "double_down")
    assert dd_trade.prev_size == 1 and dd_trade.size == 2
    assert dd_trade.notional == 1  # only the added unit is transacted, not the existing one

    cost_model = CostModel(spread_bps=10)
    costs = compute_transaction_costs(trades, positions.index, cost_model)
    assert np.isclose(costs.loc[DATES7[1]], 1 * 10 / 10_000)


def test_double_down_at_cap_is_a_true_no_op_and_costs_nothing():
    signal = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0], index=pd.date_range("2020-01-01", periods=9, freq="B"))
    positions, trades = build_positions(signal, hold_days=5, overlap_rule="double_down")
    double_down_trades = [t for t in trades if t.action == "double_down"]
    assert len(double_down_trades) == 1  # day2's repeat signal is a no-op, not a second trade

    cost_model = CostModel(spread_bps=10)
    costs = compute_transaction_costs(trades, positions.index, cost_model)
    assert costs.loc[signal.index[2]] == 0


def test_higher_spread_bps_scales_cost_linearly():
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")

    cost_2bps = compute_transaction_costs(trades, positions.index, CostModel(spread_bps=2)).sum()
    cost_5bps = compute_transaction_costs(trades, positions.index, CostModel(spread_bps=5)).sum()
    cost_10bps = compute_transaction_costs(trades, positions.index, CostModel(spread_bps=10)).sum()

    assert np.isclose(cost_5bps, cost_2bps * 2.5)
    assert np.isclose(cost_10bps, cost_2bps * 5)


def test_commission_hook_defaults_to_zero_and_can_be_enabled():
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=3, overlap_rule="ignore_new_signal")

    zero_commission = compute_transaction_costs(trades, positions.index, CostModel(spread_bps=5))
    with_commission = compute_transaction_costs(
        trades, positions.index, CostModel(spread_bps=5, commission_per_trade=0.001)
    )
    assert np.isclose(with_commission.sum() - zero_commission.sum(), 0.001 * len(trades))


def test_apply_costs_net_equals_gross_minus_cost():
    signal = pd.Series([1, -1, 0, 1, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=2, overlap_rule="restart_clock")
    y_true = pd.Series(np.linspace(-0.01, 0.01, len(DATES7)), index=DATES7)

    result = apply_costs(positions, y_true, trades, CostModel(spread_bps=5))
    pd.testing.assert_series_equal(
        result["net_return"], result["gross_return"] - result["cost"], check_names=False
    )
    pd.testing.assert_series_equal(result["gross_return"], positions * y_true, check_names=False)


def test_zero_spread_means_net_equals_gross():
    signal = pd.Series([1, -1, 0, 1, 0, 0, 0], index=DATES7)
    positions, trades = build_positions(signal, hold_days=2, overlap_rule="double_down")
    y_true = pd.Series(np.linspace(-0.01, 0.01, len(DATES7)), index=DATES7)

    result = apply_costs(positions, y_true, trades, CostModel(spread_bps=0))
    pd.testing.assert_series_equal(result["net_return"], result["gross_return"], check_names=False)
