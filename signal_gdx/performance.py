import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Standard performance metrics for a daily strategy-return series."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {k: np.nan for k in ["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "hit_rate", "n_days"]}

    equity = (1.0 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    n_days = len(returns)
    annualized_return = (1.0 + total_return) ** (periods_per_year / n_days) - 1.0
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    hit_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_days": n_days,
    }


def summarize_gross_vs_net(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    cost: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Every metric computed on both gross and net returns, side by side,
    plus how much of the gross edge the costs consumed.
    """
    gross_metrics = compute_performance_metrics(gross_returns, periods_per_year)
    net_metrics = compute_performance_metrics(net_returns, periods_per_year)

    table = pd.DataFrame({"gross": gross_metrics, "net": net_metrics})
    table["cost_impact"] = table["gross"] - table["net"]

    total_cost = cost.sum()
    total_gross_return = gross_metrics["total_return"]
    cost_pct_of_gross = total_cost / abs(total_gross_return) if total_gross_return not in (0, np.nan) else np.nan
    table.loc["total_cost_paid"] = [total_cost, np.nan, np.nan]
    table.loc["cost_as_pct_of_gross_return"] = [cost_pct_of_gross, np.nan, np.nan]

    return table
