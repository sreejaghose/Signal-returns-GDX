from dataclasses import dataclass

import numpy as np
import pandas as pd

VALID_OVERLAP_RULES = ("ignore_new_signal", "restart_clock", "double_down")


def raw_threshold_signal(y_pred: pd.Series, threshold: float) -> pd.Series:
    """Long/short/flat signal from thresholding predicted returns directly.

    +1 where y_pred > threshold, -1 where y_pred < -threshold, else 0.
    NaN predictions (burn-in, embargo gaps) map to flat (0).
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    signal = pd.Series(0, index=y_pred.index, dtype=int)
    signal[y_pred > threshold] = 1
    signal[y_pred < -threshold] = -1
    return signal


def compute_prediction_zscore(y_pred: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of recent predicted returns, using only data through day t.

    z_t = (y_pred_t - rolling_mean(y_pred)_t) / rolling_std(y_pred)_t, where the
    rolling window is the `window` most recent predictions ending at t (inclusive).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    roll_mean = y_pred.rolling(window).mean()
    roll_std = y_pred.rolling(window).std()
    return (y_pred - roll_mean) / roll_std


def zscore_threshold_signal(y_pred: pd.Series, window: int, threshold: float) -> pd.Series:
    """Long/short/flat signal from thresholding the rolling z-score of predictions."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    z = compute_prediction_zscore(y_pred, window)
    signal = pd.Series(0, index=y_pred.index, dtype=int)
    signal[z > threshold] = 1
    signal[z < -threshold] = -1
    return signal


def sweep_signal_thresholds(
    y_pred: pd.Series,
    raw_thresholds: list[float],
    zscore_thresholds: list[float],
    zscore_window: int,
) -> dict[tuple[str, float], pd.Series]:
    """Build a signal series for every (method, threshold) combination.

    Returns a dict keyed by ("raw", threshold) or ("zscore", threshold),
    each value a +1/0/-1 signal series -- one entry per sweep point, so
    downstream code can run the position/PnL logic across the whole grid.
    """
    signals: dict[tuple[str, float], pd.Series] = {}
    for t in raw_thresholds:
        signals[("raw", t)] = raw_threshold_signal(y_pred, t)
    for t in zscore_thresholds:
        signals[("zscore", t)] = zscore_threshold_signal(y_pred, zscore_window, t)
    return signals


@dataclass
class Trade:
    date: pd.Timestamp
    action: str  # "open", "restart", "double_down", "flip", "close"
    direction: int
    size: int


def build_positions(
    signal: pd.Series,
    hold_days: int,
    overlap_rule: str,
) -> tuple[pd.Series, list[Trade]]:
    """Turn a +1/0/-1 entry signal into a daily position series with holding logic.

    Execution fills at the same close as the signal: a nonzero signal on day t
    opens (or modifies) a position using day t's close, and the resulting
    exposure on day t earns the return realized from day t to t+1 -- i.e. the
    returned position series is meant to be multiplied elementwise against the
    next-day return column already produced by the backtest harness, with no
    extra shift.

    A position, once open, is held for `hold_days` trading days (day t counts
    as day 1 of the hold). What happens when a *new* nonzero signal arrives
    while a position is already open (an "overlapping" signal) is controlled
    by overlap_rule:

      - "ignore_new_signal": the new signal is ignored outright, regardless of
        its direction. The existing position runs to its originally
        scheduled expiry untouched.
      - "restart_clock": the existing position is closed and a fresh one is
        opened in the new signal's direction (same direction: this simply
        extends the hold; opposite direction: it flips the position), sized
        at 1x, with the hold clock reset to hold_days from today.
      - "double_down": if the new signal agrees with the current position's
        direction, size is increased to 2x (capped there -- it does not
        stack beyond a double) without touching the hold clock. If the new
        signal is in the *opposite* direction, doubling down makes no sense,
        so it is treated like restart_clock: the position flips to the new
        direction at 1x with the clock reset.

    Args:
        signal: +1/0/-1 series indexed by date (as from raw_threshold_signal
            or zscore_threshold_signal). NaN/0 both mean "no new signal".
        hold_days: number of trading days a position is held once opened (>=1).
        overlap_rule: one of VALID_OVERLAP_RULES.

    Returns:
        (positions, trades) where positions is a signed-size series (e.g. -2,
        -1, 0, 1, 2) indexed the same as `signal`, and trades is an
        chronological log of every open/modify/close event for auditing.
    """
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if overlap_rule not in VALID_OVERLAP_RULES:
        raise ValueError(f"overlap_rule must be one of {VALID_OVERLAP_RULES}")

    signal = signal.fillna(0).astype(int)

    positions = pd.Series(0, index=signal.index, dtype=int)
    trades: list[Trade] = []

    pos_dir = 0
    pos_size = 0
    days_remaining = 0

    for date, raw_signal in signal.items():
        if days_remaining > 0:
            if raw_signal != 0:
                if overlap_rule == "ignore_new_signal":
                    pass
                elif overlap_rule == "restart_clock":
                    action = "restart" if raw_signal == pos_dir else "flip"
                    pos_dir, pos_size, days_remaining = raw_signal, 1, hold_days
                    trades.append(Trade(date, action, pos_dir, pos_size))
                elif overlap_rule == "double_down":
                    if raw_signal == pos_dir:
                        pos_size = min(pos_size + 1, 2)
                        trades.append(Trade(date, "double_down", pos_dir, pos_size))
                    else:
                        pos_dir, pos_size, days_remaining = raw_signal, 1, hold_days
                        trades.append(Trade(date, "flip", pos_dir, pos_size))

            positions.loc[date] = pos_dir * pos_size
            days_remaining -= 1
            if days_remaining == 0:
                trades.append(Trade(date, "close", 0, 0))
                pos_dir, pos_size = 0, 0
        else:
            if raw_signal != 0:
                pos_dir, pos_size, days_remaining = raw_signal, 1, hold_days
                trades.append(Trade(date, "open", pos_dir, pos_size))
                positions.loc[date] = pos_dir * pos_size
                days_remaining -= 1
                if days_remaining == 0:
                    trades.append(Trade(date, "close", 0, 0))
                    pos_dir, pos_size = 0, 0

    return positions, trades
