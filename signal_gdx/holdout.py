import pandas as pd

from .costs import CostModel
from .evaluation import evaluate_parameter_combination
from .signals import raw_threshold_signal, zscore_threshold_signal

# A candidate's holdout Sharpe is flagged "likely overfit despite passing the
# robustness check" using the same conceptual rule as the local robustness
# check: losing at least half the in-sample Sharpe, or a solidly-positive
# in-sample Sharpe collapsing to zero or below.
DEGRADATION_RELATIVE_DROP = 0.5
SOLID_IN_SAMPLE_FLOOR = 0.3


def compute_holdout_start(full_index: pd.DatetimeIndex, holdout_fraction: float) -> pd.Timestamp:
    """The date marking the start of the last `holdout_fraction` of the full
    chronological span (by calendar time, not row count), snapped forward to
    the nearest date actually present in `full_index`.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    start, end = full_index.min(), full_index.max()
    total_days = (end - start).days
    cutoff = end - pd.Timedelta(days=total_days * holdout_fraction)
    return full_index[full_index >= cutoff].min()


def evaluate_candidate_on_holdout(
    y_pred: pd.Series,
    y_true: pd.Series,
    method: str,
    threshold: float,
    lookback: int,
    hold_days: int,
    overlap_rule: str,
    cost_model: CostModel,
    holdout_start: pd.Timestamp,
) -> dict:
    """Run one frozen (already-selected) parameter combination, unmodified,
    on the chronological tail from `holdout_start` onward.

    The signal/position/cost logic is identical to how the candidate was
    originally scored -- nothing is refit or re-tuned here. Each day's
    prediction in this tail was itself produced by a walk-forward model that
    only ever saw data strictly before that day (verified elsewhere in this
    project), so there is no row-level lookahead. The caveat worth stating
    plainly: the *selection* of this parameter combination, in the grid
    search and robustness check, used aggregate performance over the full
    OOS period -- which includes this same tail. This check is therefore a
    consistency read on the frozen parameters' most recent behavior, not a
    textbook clean holdout that never contributed to selection; a fully
    clean version would require re-running the grid search and robustness
    check with the holdout excluded from the start.
    """
    holdout_pred = y_pred[y_pred.index >= holdout_start]
    holdout_true = y_true[y_true.index >= holdout_start]

    if method == "raw":
        signal = raw_threshold_signal(holdout_pred, threshold)
    else:
        signal = zscore_threshold_signal(holdout_pred, lookback, threshold)

    return evaluate_parameter_combination(
        holdout_pred, holdout_true, signal, hold_days=hold_days, overlap_rule=overlap_rule, cost_model=cost_model
    )


def flag_holdout_degradation(in_sample_sharpe: float, holdout_sharpe: float) -> dict:
    """Whether a candidate's holdout Sharpe is drastically worse than its
    in-sample Sharpe -- likely overfit despite having passed an earlier
    robustness check, which only tests neighboring *parameters*, not a
    genuinely different time period.
    """
    if in_sample_sharpe == 0:
        relative_drop = float("nan")
    else:
        relative_drop = (in_sample_sharpe - holdout_sharpe) / abs(in_sample_sharpe)

    is_degraded = bool(
        (not pd.isna(relative_drop) and relative_drop >= DEGRADATION_RELATIVE_DROP)
        or (in_sample_sharpe > SOLID_IN_SAMPLE_FLOOR and holdout_sharpe <= 0)
    )

    return {
        "relative_drop": relative_drop,
        "flagged_overfit": is_degraded,
    }
