from dataclasses import dataclass

import pandas as pd

from .signals import Trade


@dataclass
class CostModel:
    """Transaction cost parameters applied at market-on-close execution.

    spread_bps: spread/slippage cost per unit of exposure transacted, in
        basis points of capital (e.g. 5 = 5bps). Applied to every trade
        event's notional (see Trade.notional / build_positions).
    commission_per_trade: flat cost per logged Trade event, as a fraction
        of allocated capital, independent of size. Defaults to 0 -- this
        is a bps-only cost model unless a commission is explicitly
        supplied; the hook is here for a fixed-fee or per-share model.
    """

    spread_bps: float
    commission_per_trade: float = 0.0


def compute_transaction_costs(trades: list[Trade], index: pd.DatetimeIndex, cost_model: CostModel) -> pd.Series:
    """Per-date transaction cost series, as a fraction of allocated capital.

    Every logged Trade event is charged -- entries, exits, restarts, flips,
    and the incremental unit added by a double_down -- not just entry/exit
    pairs, since a restart or double-down is itself a real transaction
    (Trade.notional already encodes exactly how many exposure units each
    event actually transacted; see build_positions). Cost per event is
    `notional * spread_bps/10000` (spread/slippage) plus a flat
    `commission_per_trade` (0 by default -- the hook for a non-bps cost).

    Multiple events on the same date (e.g. a same-day flip) accumulate.
    """
    costs = pd.Series(0.0, index=index)
    spread_frac = cost_model.spread_bps / 10_000.0
    for trade in trades:
        costs.loc[trade.date] += trade.notional * spread_frac + cost_model.commission_per_trade
    return costs


def apply_costs(
    positions: pd.Series,
    y_true: pd.Series,
    trades: list[Trade],
    cost_model: CostModel,
) -> pd.DataFrame:
    """Gross and net-of-cost daily strategy returns, side by side.

    Costs are applied at market-on-close execution: charged on the same
    date the triggering Trade event fills, against the same capital base
    used for the gross return (positions * next-day realized return).
    """
    gross_return = positions * y_true
    cost = compute_transaction_costs(trades, positions.index, cost_model)
    net_return = gross_return - cost
    return pd.DataFrame(
        {
            "position": positions,
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
        }
    )
