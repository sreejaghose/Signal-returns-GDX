from collections import defaultdict

import numpy as np
import pandas as pd

from .costs import CostModel, apply_costs
from .performance import (
    TRADING_DAYS_PER_YEAR,
    TRADE_INITIATION_ACTIONS,
    compute_performance_metrics,
    compute_trades_per_month,
)
from .signals import Trade, build_positions, sweep_signal_thresholds

VALID_OVERLAP_RULES_DEFAULT = ("ignore_new_signal", "restart_clock", "double_down")


def _trade_episodes(index: pd.DatetimeIndex, trades: list[Trade]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Reconstruct each trade's (start_date, end_date) span from the trade log.

    A new episode begins at every "open"/"restart"/"flip" event -- the same
    events counted as trades by compute_trades_per_month. It ends at the
    "close" event that follows (if the hold expired naturally before any
    later start), otherwise it's cut short by the very next start event
    (an overlapping restart or flip), otherwise -- if the data simply ran
    out mid-hold -- it ends at the last available date.
    """
    starts = [t for t in trades if t.action in TRADE_INITIATION_ACTIONS]
    close_dates = sorted(t.date for t in trades if t.action == "close")

    episodes = []
    for i, start_trade in enumerate(starts):
        start_date = start_trade.date
        next_start_date = starts[i + 1].date if i + 1 < len(starts) else None

        candidate_closes = [
            d for d in close_dates if d > start_date and (next_start_date is None or d < next_start_date)
        ]
        if candidate_closes:
            end_date = candidate_closes[0]
        elif next_start_date is not None:
            end_date = index[index.get_loc(next_start_date) - 1]
        else:
            end_date = index[-1]

        episodes.append((start_date, end_date))

    return episodes


def compute_hold_day_decay(
    index: pd.DatetimeIndex,
    trades: list[Trade],
    gross_return: pd.Series,
    net_return: pd.Series,
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    """Average return contributed by day 1, day 2, ... of a trade, isolated.

    Each trade episode's daily returns are indexed 1..len(episode) from its
    own start (a restart/flip starts a fresh count; a double_down does not,
    since it doesn't begin a new episode). Day k's value is the average,
    across every episode that lasted at least k days, of that k-th day's own
    return -- not a cumulative return -- so the curve shows whether later
    days in a hold tend to add or subtract relative to earlier ones. The
    episode count per day is returned alongside so a thin tail (few episodes
    ever held that long) is visible rather than silently averaged away.
    """
    episodes = _trade_episodes(index, trades)

    gross_by_day: dict[int, list[float]] = defaultdict(list)
    net_by_day: dict[int, list[float]] = defaultdict(list)

    for start_date, end_date in episodes:
        start_pos = index.get_loc(start_date)
        end_pos = index.get_loc(end_date)
        for day_idx, pos in enumerate(range(start_pos, end_pos + 1), start=1):
            date = index[pos]
            gross_by_day[day_idx].append(gross_return.loc[date])
            net_by_day[day_idx].append(net_return.loc[date])

    decay_gross = {day: float(np.mean(values)) for day, values in sorted(gross_by_day.items())}
    decay_net = {day: float(np.mean(values)) for day, values in sorted(net_by_day.items())}
    episode_counts = {day: len(values) for day, values in sorted(gross_by_day.items())}
    return decay_gross, decay_net, episode_counts


def evaluate_parameter_combination(
    y_pred: pd.Series,
    y_true: pd.Series,
    signal: pd.Series,
    hold_days: int,
    overlap_rule: str,
    cost_model: CostModel,
    min_trades_per_month: float = 1.0,
) -> dict:
    """Full evaluation of one signal + hold_days + overlap_rule + cost_model
    combination, returned as a single flat dict (one row) so results can be
    collected into a DataFrame and aggregated across a full parameter grid.

    Includes gross and net-of-cost Sharpe, hit rate, average return per
    trade, max drawdown, turnover, trades-per-month (with the
    meets_min_frequency flag), and a day-of-hold decay curve isolating the
    average return contributed by each day of a trade's life, both gross
    and net of costs.
    """
    positions, trades = build_positions(signal, hold_days, overlap_rule)
    result = apply_costs(positions, y_true, trades, cost_model)
    gross_return, net_return, cost = result["gross_return"], result["net_return"], result["cost"]

    gross_metrics = compute_performance_metrics(gross_return)
    net_metrics = compute_performance_metrics(net_return)
    frequency = compute_trades_per_month(trades, positions.index, min_trades_per_month)
    n_trades = frequency["n_trades"]

    avg_gross_return_per_trade = gross_return.sum() / n_trades if n_trades > 0 else np.nan
    avg_net_return_per_trade = net_return.sum() / n_trades if n_trades > 0 else np.nan

    total_notional_traded = sum(t.notional for t in trades)
    n_years = len(positions.index) / TRADING_DAYS_PER_YEAR
    annual_turnover = total_notional_traded / n_years if n_years > 0 else np.nan

    decay_gross, decay_net, decay_episode_counts = compute_hold_day_decay(
        positions.index, trades, gross_return, net_return
    )

    return {
        "hold_days": hold_days,
        "overlap_rule": overlap_rule,
        "spread_bps": cost_model.spread_bps,
        "commission_per_trade": cost_model.commission_per_trade,
        "n_trades": n_trades,
        "trades_per_month": frequency["trades_per_month"],
        "meets_min_frequency": frequency["meets_min_frequency"],
        "gross_sharpe": gross_metrics["sharpe"],
        "net_sharpe": net_metrics["sharpe"],
        "gross_hit_rate": gross_metrics["hit_rate"],
        "net_hit_rate": net_metrics["hit_rate"],
        "avg_gross_return_per_trade": avg_gross_return_per_trade,
        "avg_net_return_per_trade": avg_net_return_per_trade,
        "gross_total_return": gross_metrics["total_return"],
        "net_total_return": net_metrics["total_return"],
        "gross_max_drawdown": gross_metrics["max_drawdown"],
        "net_max_drawdown": net_metrics["max_drawdown"],
        "total_notional_traded": total_notional_traded,
        "annual_turnover": annual_turnover,
        "total_cost_paid": cost.sum(),
        "hold_day_decay_gross": decay_gross,
        "hold_day_decay_net": decay_net,
        "hold_day_decay_episode_counts": decay_episode_counts,
    }


def evaluate_signal_grid(
    y_pred: pd.Series,
    y_true: pd.Series,
    raw_thresholds: list[float],
    zscore_thresholds: list[float],
    zscore_window: int,
    hold_days_grid: list[int],
    cost_model: CostModel,
    overlap_rules: list[str] = VALID_OVERLAP_RULES_DEFAULT,
    min_trades_per_month: float = 1.0,
) -> pd.DataFrame:
    """Run every (signal method x threshold x hold_days x overlap_rule)
    combination through evaluate_parameter_combination and return one row
    of results per combination.

    Every row reports trades_per_month and meets_min_frequency alongside the
    rest of the metrics, so a parameter combination that only traded a
    handful of times over the whole backtest -- and might otherwise show an
    inflated Sharpe from a few lucky trades -- is flagged rather than
    silently mixed in with combinations that traded often enough to be
    operationally practical. Rows are not dropped; meets_min_frequency=False
    just marks them for exclusion by the caller (e.g. `results[results.meets_min_frequency]`).

    Args:
        y_pred: OOS predicted returns (as from a walk_forward_backtest predictions frame).
        y_true: realized next-day returns aligned to y_pred's index.
        raw_thresholds, zscore_thresholds, zscore_window: passed to sweep_signal_thresholds.
        hold_days_grid: hold_days values to sweep.
        cost_model: CostModel applied uniformly across the grid.
        overlap_rules: overlap_rule values to sweep.
        min_trades_per_month: threshold for meets_min_frequency (default 1/month).

    Returns:
        DataFrame, one row per combination -- see evaluate_parameter_combination
        for the full column set, plus "method" and "threshold" identifying
        the signal used.
    """
    signal_grid = sweep_signal_thresholds(y_pred, raw_thresholds, zscore_thresholds, zscore_window)

    rows = []
    for (method, threshold), signal in signal_grid.items():
        for hold_days in hold_days_grid:
            for overlap_rule in overlap_rules:
                row = evaluate_parameter_combination(
                    y_pred, y_true, signal, hold_days, overlap_rule, cost_model, min_trades_per_month
                )
                rows.append({"method": method, "threshold": threshold, **row})

    return pd.DataFrame(rows)
