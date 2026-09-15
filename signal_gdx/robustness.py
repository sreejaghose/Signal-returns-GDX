import pandas as pd

STEPS = [-2, -1, 0, 1, 2]
PARAMETERS = ["lookback", "threshold", "hold_days", "reg_strength"]

LOOKBACK_STEP = 5
LOOKBACK_MIN = 5
HOLD_DAYS_STEP = 2
HOLD_DAYS_MIN = 1
REG_STRENGTH_MULTIPLICATIVE_STEP = 2.0
RAW_THRESHOLD_STEP = 0.0005
ZSCORE_THRESHOLD_STEP = 0.25

# A candidate is "fragile" if a single *one-step* nudge -- the smallest
# perturbation checked -- costs at least this fraction of its baseline net
# Sharpe. Two-step nudges are reported for context (do the neighbors keep
# decaying gradually, or does it accelerate?) but never trigger the label on
# their own: the ask is specifically about collapse from a small nudge.
FRAGILE_RELATIVE_DROP_AT_1_STEP = 0.5


def nudge_lookback(base: int, step: int) -> int | None:
    value = base + step * LOOKBACK_STEP
    return value if value >= LOOKBACK_MIN else None


def nudge_hold_days(base: int, step: int) -> int | None:
    value = base + step * HOLD_DAYS_STEP
    return value if value >= HOLD_DAYS_MIN else None


def nudge_reg_strength(base: float, step: int) -> float:
    return base * (REG_STRENGTH_MULTIPLICATIVE_STEP**step)


def nudge_threshold(base: float, method: str, step: int) -> float | None:
    step_size = RAW_THRESHOLD_STEP if method == "raw" else ZSCORE_THRESHOLD_STEP
    value = round(base + step * step_size, 10)
    return value if value >= 0 else None


def build_nudge_plan(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per (candidate, parameter, step) with the full parameter
    tuple to evaluate: exactly one parameter is nudged away from the
    candidate's own value, holding lookback/threshold/hold_days/reg_strength
    (whichever three aren't the target) plus method/overlap_rule/spread_bps
    fixed at the candidate's baseline.

    `candidates` must have columns: lookback, model_type, reg_strength,
    method, threshold, hold_days, overlap_rule, spread_bps, net_sharpe,
    net_total_return (the last two become each row's baseline_* columns).
    A row where the nudge falls out of valid range (e.g. hold_days < 1, a
    negative threshold) is still emitted with valid=False and no nudged
    value, so a candidate's grid stays visibly 5-wide even where an edge is
    unreachable.
    """
    rows = []
    for candidate_id, candidate in candidates.reset_index(drop=True).iterrows():
        base = {
            "lookback": candidate["lookback"],
            "model_type": candidate["model_type"],
            "reg_strength": candidate["reg_strength"],
            "method": candidate["method"],
            "threshold": candidate["threshold"],
            "hold_days": candidate["hold_days"],
            "overlap_rule": candidate["overlap_rule"],
            "spread_bps": candidate["spread_bps"],
        }
        for parameter in PARAMETERS:
            for step in STEPS:
                tup = dict(base)
                nudged_value = None
                valid = True

                if parameter == "lookback":
                    nudged_value = nudge_lookback(base["lookback"], step)
                    valid = nudged_value is not None
                    if valid:
                        tup["lookback"] = nudged_value
                elif parameter == "hold_days":
                    nudged_value = nudge_hold_days(base["hold_days"], step)
                    valid = nudged_value is not None
                    if valid:
                        tup["hold_days"] = nudged_value
                elif parameter == "reg_strength":
                    nudged_value = nudge_reg_strength(base["reg_strength"], step)
                    tup["reg_strength"] = nudged_value
                elif parameter == "threshold":
                    nudged_value = nudge_threshold(base["threshold"], base["method"], step)
                    valid = nudged_value is not None
                    if valid:
                        tup["threshold"] = nudged_value

                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "parameter": parameter,
                        "step": step,
                        "nudged_value": nudged_value,
                        "valid": valid,
                        "baseline_net_sharpe": candidate["net_sharpe"],
                        "baseline_net_total_return": candidate["net_total_return"],
                        **tup,
                    }
                )
    return pd.DataFrame(rows)


def expensive_combos_needed(nudge_plan: pd.DataFrame) -> list[tuple[int, str, float]]:
    """Distinct (lookback, model_type, reg_strength) triples the nudge plan
    references -- only "lookback" and "reg_strength" nudges ever change
    this triple; "threshold" and "hold_days" nudges reuse the candidate's
    own (already-cached) predictions."""
    relevant = nudge_plan[nudge_plan["valid"] & nudge_plan["parameter"].isin(["lookback", "reg_strength"])]
    triples = relevant[["lookback", "model_type", "reg_strength"]].drop_duplicates()
    return list(triples.itertuples(index=False, name=None))


def compute_robustness_verdicts(results: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate_id: verdict, worst relative drop seen at a
    1-step and at a 2-step nudge, and which (parameter, step) triggered a
    fragile verdict, if any.

    `results` must have the same rows as build_nudge_plan's output plus a
    filled-in `net_sharpe` column (NaN/absent for invalid rows).
    """
    verdict_rows = []
    for candidate_id, group in results.groupby("candidate_id"):
        baseline = group["baseline_net_sharpe"].iloc[0]
        group = group[group["valid"]]

        def relative_drop(net_sharpe):
            if baseline == 0:
                return float("nan")
            return (baseline - net_sharpe) / abs(baseline)

        one_step = group[group["step"].isin([-1, 1])].copy()
        two_step = group[group["step"].isin([-2, 2])].copy()
        one_step["relative_drop"] = one_step["net_sharpe"].apply(relative_drop)
        two_step["relative_drop"] = two_step["net_sharpe"].apply(relative_drop)

        worst_one_step = one_step.loc[one_step["relative_drop"].idxmax()] if len(one_step) else None
        worst_two_step = two_step.loc[two_step["relative_drop"].idxmax()] if len(two_step) else None

        is_fragile = worst_one_step is not None and (
            worst_one_step["relative_drop"] >= FRAGILE_RELATIVE_DROP_AT_1_STEP
            or (baseline > 0.3 and worst_one_step["net_sharpe"] <= 0)
        )

        verdict_rows.append(
            {
                "candidate_id": candidate_id,
                "baseline_net_sharpe": baseline,
                "verdict": "fragile" if is_fragile else "robust",
                "worst_one_step_relative_drop": worst_one_step["relative_drop"] if worst_one_step is not None else float("nan"),
                "worst_one_step_parameter": worst_one_step["parameter"] if worst_one_step is not None else None,
                "worst_one_step_direction": int(worst_one_step["step"]) if worst_one_step is not None else None,
                "worst_two_step_relative_drop": worst_two_step["relative_drop"] if worst_two_step is not None else float("nan"),
                "worst_two_step_parameter": worst_two_step["parameter"] if worst_two_step is not None else None,
            }
        )
    return pd.DataFrame(verdict_rows)
