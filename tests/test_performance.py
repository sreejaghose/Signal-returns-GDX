import numpy as np
import pandas as pd

from signal_gdx import compute_performance_metrics, summarize_gross_vs_net


def test_metrics_on_known_constant_return_series():
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = pd.Series(0.001, index=dates)  # constant positive daily return, zero vol
    metrics = compute_performance_metrics(returns, periods_per_year=252)

    expected_total = 1.001**252 - 1
    assert np.isclose(metrics["total_return"], expected_total)
    assert np.isclose(metrics["annualized_return"], expected_total, atol=1e-9)
    assert np.isclose(metrics["annualized_vol"], 0, atol=1e-12)
    assert metrics["max_drawdown"] == 0
    assert metrics["hit_rate"] == 1.0
    assert metrics["n_days"] == 252


def test_metrics_max_drawdown_on_known_path():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    # equity path: 1 -> 1.10 -> 0.99 -> 1.05 ; drawdown from peak 1.10 to trough 0.99 = -10%
    returns = pd.Series([0.10, -0.10, 1.05 / 0.99 - 1, 0.0], index=dates)
    metrics = compute_performance_metrics(returns)
    assert np.isclose(metrics["max_drawdown"], -0.10)


def test_empty_returns_series_does_not_crash():
    returns = pd.Series([np.nan, np.nan], index=pd.date_range("2020-01-01", periods=2))
    metrics = compute_performance_metrics(returns)
    assert all(np.isnan(v) for v in metrics.values())


def test_summarize_gross_vs_net_cost_impact_matches_difference():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.normal(0.0005, 0.01, size=100), index=dates)
    cost = pd.Series(0.0002, index=dates)
    net = gross - cost

    table = summarize_gross_vs_net(gross, net, cost)

    assert np.isclose(table.loc["sharpe", "gross"] - table.loc["sharpe", "net"], table.loc["sharpe", "cost_impact"])
    assert np.isclose(table.loc["total_return", "cost_impact"], table.loc["total_return", "gross"] - table.loc["total_return", "net"])
    assert np.isclose(table.loc["total_cost_paid", "gross"], cost.sum())
    # net metrics should be strictly worse than gross when costs are positive
    assert table.loc["sharpe", "net"] < table.loc["sharpe", "gross"]
    assert table.loc["total_return", "net"] < table.loc["total_return", "gross"]


def test_summarize_gross_vs_net_zero_cost_means_identical_metrics():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    rng = np.random.default_rng(1)
    gross = pd.Series(rng.normal(0.0003, 0.008, size=50), index=dates)
    cost = pd.Series(0.0, index=dates)
    net = gross.copy()

    table = summarize_gross_vs_net(gross, net, cost)
    for metric in ["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "hit_rate"]:
        assert np.isclose(table.loc[metric, "gross"], table.loc[metric, "net"])
        assert np.isclose(table.loc[metric, "cost_impact"], 0.0)
