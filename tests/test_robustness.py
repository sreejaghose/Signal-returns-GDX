import numpy as np
import pandas as pd
import pytest

from signal_gdx.robustness import (
    build_nudge_plan,
    compute_robustness_verdicts,
    expensive_combos_needed,
    nudge_hold_days,
    nudge_lookback,
    nudge_reg_strength,
    nudge_threshold,
)


def make_candidates(**overrides):
    base = {
        "lookback": 20,
        "model_type": "elasticnet",
        "reg_strength": 0.0005,
        "method": "raw",
        "threshold": 0.002,
        "hold_days": 5,
        "overlap_rule": "restart_clock",
        "spread_bps": 2,
        "net_sharpe": 0.88,
        "net_total_return": 2.43,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_nudge_functions_basic_arithmetic():
    assert nudge_lookback(20, 1) == 25
    assert nudge_lookback(20, -1) == 15
    assert nudge_hold_days(5, 2) == 9
    assert nudge_reg_strength(0.0005, 1) == pytest.approx(0.001)
    assert nudge_reg_strength(0.0005, -1) == pytest.approx(0.00025)
    assert nudge_threshold(0.002, "raw", 1) == pytest.approx(0.0025)
    assert nudge_threshold(0.5, "zscore", -1) == pytest.approx(0.25)


def test_nudge_functions_reject_out_of_range():
    assert nudge_lookback(10, -2) is None  # 10 - 10 = 0 < min 5
    assert nudge_hold_days(1, -1) is None  # 1 - 2 = -1 < min 1
    assert nudge_threshold(0.0005, "raw", -2) is None  # 0.0005 - 0.001 < 0


def test_build_nudge_plan_has_five_steps_per_parameter_per_candidate():
    candidates = make_candidates()
    plan = build_nudge_plan(candidates)
    assert len(plan) == 4 * 5  # 4 parameters x 5 steps, 1 candidate
    for parameter in ["lookback", "threshold", "hold_days", "reg_strength"]:
        steps = sorted(plan[plan["parameter"] == parameter]["step"].tolist())
        assert steps == [-2, -1, 0, 1, 2]


def test_build_nudge_plan_step_zero_equals_baseline_for_every_parameter():
    candidates = make_candidates()
    plan = build_nudge_plan(candidates)
    zero_rows = plan[plan["step"] == 0]
    for _, row in zero_rows.iterrows():
        assert row["lookback"] == 20
        assert row["threshold"] == 0.002
        assert row["hold_days"] == 5
        assert row["reg_strength"] == pytest.approx(0.0005)
        assert row["valid"]


def test_build_nudge_plan_only_targets_one_parameter_at_a_time():
    candidates = make_candidates()
    plan = build_nudge_plan(candidates)
    lookback_row = plan[(plan["parameter"] == "lookback") & (plan["step"] == 1)].iloc[0]
    assert lookback_row["lookback"] == 25
    assert lookback_row["threshold"] == 0.002  # untouched
    assert lookback_row["hold_days"] == 5  # untouched
    assert lookback_row["reg_strength"] == pytest.approx(0.0005)  # untouched


def test_build_nudge_plan_marks_out_of_range_as_invalid_but_keeps_the_row():
    candidates = make_candidates(hold_days=1)  # step -2 -> hold_days -3, invalid
    plan = build_nudge_plan(candidates)
    invalid_row = plan[(plan["parameter"] == "hold_days") & (plan["step"] == -2)].iloc[0]
    assert bool(invalid_row["valid"]) is False
    assert pd.isna(invalid_row["nudged_value"])
    # still present so the 5-wide grid is visible, just not evaluable
    assert len(plan[plan["parameter"] == "hold_days"]) == 5


def test_expensive_combos_needed_only_covers_lookback_and_reg_strength():
    candidates = make_candidates()
    plan = build_nudge_plan(candidates)
    combos = expensive_combos_needed(plan)
    # lookback nudges: 10, 15, 20(base), 25, 30 -> 5 combos (all elasticnet/0.0005)
    # reg_strength nudges: 0.000125, 0.00025, 0.0005(base), 0.001, 0.002 -> 5 combos (all L=20)
    # the step=0 triple (L=20, reg=0.0005) is shared between both sweeps, so 5+5-1 unique
    assert len(combos) == 5 + 5 - 1
    for lookback, model_type, reg_strength in combos:
        assert model_type == "elasticnet"
    # threshold/hold_days nudges must NOT introduce new triples
    lookbacks_from_lookback_nudges = {c[0] for c in combos if c[2] == pytest.approx(0.0005)}
    assert lookbacks_from_lookback_nudges == {10, 15, 20, 25, 30}


def test_expensive_combos_needed_dedupes_across_candidates():
    candidates = pd.concat(
        [make_candidates(lookback=20, reg_strength=0.0005), make_candidates(lookback=20, reg_strength=0.0005)],
        ignore_index=True,
    )
    plan = build_nudge_plan(candidates)
    combos = expensive_combos_needed(plan)
    # identical candidates should produce identical nudge combos, deduped
    assert len(combos) == len(set(combos))


def _fill_synthetic_net_sharpe(plan: pd.DataFrame, drop_map: dict) -> pd.DataFrame:
    """drop_map: {(parameter, step): net_sharpe} overrides; anything else
    defaults to a mild linear decay from baseline."""
    plan = plan.copy()
    plan["net_sharpe"] = np.nan
    for idx, row in plan.iterrows():
        if not row["valid"]:
            continue
        key = (row["parameter"], row["step"])
        if key in drop_map:
            plan.loc[idx, "net_sharpe"] = drop_map[key]
        elif row["step"] == 0:
            plan.loc[idx, "net_sharpe"] = row["baseline_net_sharpe"]
        else:
            plan.loc[idx, "net_sharpe"] = row["baseline_net_sharpe"] - 0.02 * abs(row["step"])
    return plan


def test_robust_candidate_labeled_robust_on_gradual_decay():
    candidates = make_candidates(net_sharpe=0.8)
    plan = build_nudge_plan(candidates)
    filled = _fill_synthetic_net_sharpe(plan, drop_map={})  # pure gradual decay everywhere
    verdicts = compute_robustness_verdicts(filled)
    assert verdicts.iloc[0]["verdict"] == "robust"


def test_fragile_candidate_labeled_fragile_on_one_step_collapse():
    candidates = make_candidates(net_sharpe=0.8)
    plan = build_nudge_plan(candidates)
    # a single one-step neighbor (lookback, +1) collapses to near zero
    filled = _fill_synthetic_net_sharpe(plan, drop_map={("lookback", 1): 0.05})
    verdicts = compute_robustness_verdicts(filled)
    row = verdicts.iloc[0]
    assert row["verdict"] == "fragile"
    assert row["worst_one_step_parameter"] == "lookback"
    assert row["worst_one_step_direction"] == 1


def test_two_step_only_collapse_does_not_trigger_fragile_alone():
    """A collapse that only shows up at step=+/-2 (not +/-1) should not, by
    itself, mark the candidate fragile -- the ask is about small nudges."""
    candidates = make_candidates(net_sharpe=0.8)
    plan = build_nudge_plan(candidates)
    filled = _fill_synthetic_net_sharpe(plan, drop_map={("threshold", 2): -0.5})
    verdicts = compute_robustness_verdicts(filled)
    row = verdicts.iloc[0]
    assert row["verdict"] == "robust"
    assert row["worst_two_step_parameter"] == "threshold"


def test_sign_flip_to_zero_from_solid_positive_baseline_is_fragile_even_without_50pct_math():
    candidates = make_candidates(net_sharpe=0.5)
    plan = build_nudge_plan(candidates)
    # exactly 50% drop threshold edge case: net_sharpe hits 0 (>=50% relative drop AND sign floor)
    filled = _fill_synthetic_net_sharpe(plan, drop_map={("hold_days", -1): 0.0})
    verdicts = compute_robustness_verdicts(filled)
    assert verdicts.iloc[0]["verdict"] == "fragile"
