import numpy as np
import pandas as pd

from .signals import Trade

TRADING_DAYS_PER_YEAR = 252
AVG_DAYS_PER_MONTH = 365.25 / 12
TRADE_INITIATION_ACTIONS = ("open", "restart", "flip")


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


def compute_trades_per_month(trades: list[Trade], index: pd.DatetimeIndex, min_trades_per_month: float = 1.0) -> dict:
    """Trading frequency for one parameter combination, plus the min-frequency flag.

    Counts only "open", "restart", and "flip" events as trades: each is the
    strategy initiating a fresh directional stance (from flat, or reversing
    an existing one). A "close" is a scheduled, predictable unwind rather
    than a new trading decision, and a "double_down" merely sizes up an
    already-open trade -- neither is counted as a separate trade here, even
    though both are still charged transaction costs elsewhere. This keeps
    the frequency check about how often the signal actually asks you to act,
    which is what makes a sparse signal operationally impractical.

    The backtest span is measured over `index` (the full predicted/aligned
    date range the strategy was run over, not just the dates with trades),
    so a combination with zero trades still gets a real (zero) frequency
    rather than an undefined one.
    """
    n_trades = sum(1 for t in trades if t.action in TRADE_INITIATION_ACTIONS)

    if len(index) < 2:
        n_months = np.nan
    else:
        span_days = (index[-1] - index[0]).days
        n_months = span_days / AVG_DAYS_PER_MONTH

    trades_per_month = n_trades / n_months if n_months and n_months > 0 else 0.0

    return {
        "n_trades": n_trades,
        "trades_per_month": trades_per_month,
        "meets_min_frequency": trades_per_month >= min_trades_per_month,
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
