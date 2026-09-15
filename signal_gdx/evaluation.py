import pandas as pd

from .costs import CostModel, apply_costs
from .performance import compute_performance_metrics, compute_trades_per_month
from .signals import build_positions, sweep_signal_thresholds

VALID_OVERLAP_RULES_DEFAULT = ("ignore_new_signal", "restart_clock", "double_down")


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
    combination through the position/cost/performance pipeline and return
    one row of results per combination.

    Every row reports trades_per_month and meets_min_frequency alongside the
    usual gross/net metrics, so a parameter combination that only traded a
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
        DataFrame, one row per combination, columns: method, threshold,
        hold_days, overlap_rule, n_trades, trades_per_month,
        meets_min_frequency, gross_total_return, net_total_return,
        gross_sharpe, net_sharpe, gross_max_drawdown, net_max_drawdown,
        gross_hit_rate, net_hit_rate, total_cost_paid.
    """
    signal_grid = sweep_signal_thresholds(y_pred, raw_thresholds, zscore_thresholds, zscore_window)

    rows = []
    for (method, threshold), signal in signal_grid.items():
        for hold_days in hold_days_grid:
            for overlap_rule in overlap_rules:
                positions, trades = build_positions(signal, hold_days, overlap_rule)
                result = apply_costs(positions, y_true, trades, cost_model)

                gross_metrics = compute_performance_metrics(result["gross_return"])
                net_metrics = compute_performance_metrics(result["net_return"])
                frequency = compute_trades_per_month(trades, positions.index, min_trades_per_month)

                rows.append(
                    {
                        "method": method,
                        "threshold": threshold,
                        "hold_days": hold_days,
                        "overlap_rule": overlap_rule,
                        "n_trades": frequency["n_trades"],
                        "trades_per_month": frequency["trades_per_month"],
                        "meets_min_frequency": frequency["meets_min_frequency"],
                        "gross_total_return": gross_metrics["total_return"],
                        "net_total_return": net_metrics["total_return"],
                        "gross_sharpe": gross_metrics["sharpe"],
                        "net_sharpe": net_metrics["sharpe"],
                        "gross_max_drawdown": gross_metrics["max_drawdown"],
                        "net_max_drawdown": net_metrics["max_drawdown"],
                        "gross_hit_rate": gross_metrics["hit_rate"],
                        "net_hit_rate": net_metrics["hit_rate"],
                        "total_cost_paid": result["cost"].sum(),
                    }
                )

    return pd.DataFrame(rows)
