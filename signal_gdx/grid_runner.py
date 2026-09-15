import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import walk_forward_backtest
from .costs import CostModel
from .data import load_prices
from .evaluation import evaluate_signal_grid
from .features import TARGET_COL, build_features
from .models import make_shallow_lightgbm

# reg_strength is a single scalar knob, comparable across model types, but
# each model type needs its own sensible range: ElasticNet's alpha and
# LightGBM's reg_alpha/reg_lambda don't live on the same scale.
DEFAULT_REG_STRENGTH_GRID = {
    "elasticnet": [0.0005, 0.005, 0.02],
    "lightgbm": [0.5, 2.0, 5.0],
}


def make_elasticnet_fixed_model(alpha: float, l1_ratio: float = 0.5, random_state: int = 42) -> Pipeline:
    """Plain ElasticNet at a fixed alpha/l1_ratio (no internal CV search).

    Used for the grid sweep, where regularization strength is itself an
    outer grid parameter we want to compare directly -- unlike
    make_elasticnet_model()'s ElasticNetCV, which searches for its own
    alpha/l1_ratio and would make "regularization strength" meaningless as
    a separate axis.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=random_state)),
        ]
    )


def build_model_for_grid(model_type: str, reg_strength: float):
    if model_type == "elasticnet":
        return make_elasticnet_fixed_model(alpha=reg_strength)
    if model_type == "lightgbm":
        return make_shallow_lightgbm(reg_alpha=reg_strength, reg_lambda=reg_strength, n_jobs=1)
    raise ValueError(f"unknown model_type: {model_type!r}")


def _predictions_cache_path(cache_dir: Path, lookback: int, model_type: str, reg_strength: float) -> Path:
    return cache_dir / f"predictions_L{lookback}_{model_type}_reg{reg_strength}.csv"


def run_expensive_combo(
    data_path: str,
    lookback: int,
    model_type: str,
    reg_strength: float,
    max_lag: int,
    min_train_size: int,
    refit_freq: int,
    cache_dir: str,
) -> dict:
    """Build features at this lookback and walk-forward backtest this model
    (the expensive step: refits the estimator at every fold), then cache the
    resulting OOS predictions to disk. If the cache file already exists,
    skip straight to returning its path -- this is what lets a rerun of the
    grid pick up from disk instead of recomputing every combination.

    Runs as a unit of work in a process pool: takes only picklable
    primitives and does its own file I/O, so parallel workers never share
    state and never race on the same cache file.
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_path = _predictions_cache_path(cache_dir_path, lookback, model_type, reg_strength)

    if cache_path.exists():
        return {
            "lookback": lookback,
            "model_type": model_type,
            "reg_strength": reg_strength,
            "cache_path": str(cache_path),
            "was_cached": True,
        }

    prices = load_prices(data_path)
    feature_df = build_features(prices, max_lag=max_lag, lookback=lookback)
    feature_cols = [c for c in feature_df.columns if c != TARGET_COL]

    estimator = build_model_for_grid(model_type, reg_strength)

    predictions, _folds = walk_forward_backtest(
        feature_df,
        feature_cols,
        TARGET_COL,
        estimator,
        min_train_size=min_train_size,
        refit_freq=refit_freq,
        embargo=lookback,
        window="expanding",
    )
    predictions.to_csv(cache_path, index_label="date")

    return {
        "lookback": lookback,
        "model_type": model_type,
        "reg_strength": reg_strength,
        "cache_path": str(cache_path),
        "was_cached": False,
    }


def run_full_grid(
    data_path: str,
    lookback_grid: list[int],
    model_types: list[str],
    reg_strength_grid: dict[str, list[float]],
    raw_thresholds: list[float],
    zscore_thresholds: list[float],
    hold_days_grid: list[int],
    overlap_rules: list[str],
    cost_bps_grid: list[float],
    results_path: str,
    cache_dir: str,
    max_lag: int = 5,
    min_train_size: int = 750,
    refit_freq: int = 21,
    min_trades_per_month: float = 1.0,
    max_workers: int | None = None,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Run the full backtest across lookback L x model_type x regularization
    strength x threshold type/level x hold_days x overlap_rule x cost-bps,
    and return (and persist) a single combined results DataFrame.

    The grid splits into two tiers because of the cost difference: L x
    model_type x reg_strength each requires a fresh walk-forward backtest
    (refitting the estimator at every fold -- the expensive part), so those
    combinations are cached to disk per-combo and run in parallel across
    processes. Threshold/hold_days/overlap_rule/cost_bps only re-run the
    cheap signal/position/cost pipeline against already-computed
    predictions, so they're swept in a plain loop via evaluate_signal_grid
    for each expensive combo's predictions.

    If `results_path` already exists and force_rerun is False, it's loaded
    and returned immediately -- the whole grid is not recomputed. Otherwise,
    each expensive (L, model_type, reg_strength) combo's predictions are
    individually cached under `cache_dir`; a rerun (even after an
    interruption) skips any combo whose cache file is already there.

    Returns:
        DataFrame with one row per full parameter combination, including
        lookback, model_type, reg_strength, method, threshold, hold_days,
        overlap_rule, spread_bps, trades_per_month, meets_min_frequency,
        and every metric from evaluate_parameter_combination.
    """
    results_file = Path(results_path)
    if results_file.exists() and not force_rerun:
        return pd.read_pickle(results_file)

    expensive_combos = [
        (lookback, model_type, reg_strength)
        for lookback in lookback_grid
        for model_type in model_types
        for reg_strength in reg_strength_grid[model_type]
    ]

    max_workers = max_workers or os.cpu_count() or 1
    combo_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_expensive_combo,
                data_path,
                lookback,
                model_type,
                reg_strength,
                max_lag,
                min_train_size,
                refit_freq,
                cache_dir,
            ): (lookback, model_type, reg_strength)
            for lookback, model_type, reg_strength in expensive_combos
        }
        for future in as_completed(futures):
            combo_results.append(future.result())

    all_rows = []
    for combo in combo_results:
        predictions = pd.read_csv(combo["cache_path"], index_col="date", parse_dates=True)
        oos = predictions.dropna(subset=["y_pred"])
        y_pred, y_true = oos["y_pred"], oos["y_true"]

        for spread_bps in cost_bps_grid:
            grid_result = evaluate_signal_grid(
                y_pred,
                y_true,
                raw_thresholds=raw_thresholds,
                zscore_thresholds=zscore_thresholds,
                zscore_window=combo["lookback"],
                hold_days_grid=hold_days_grid,
                cost_model=CostModel(spread_bps=spread_bps),
                overlap_rules=overlap_rules,
                min_trades_per_month=min_trades_per_month,
            )
            grid_result.insert(0, "lookback", combo["lookback"])
            grid_result.insert(1, "model_type", combo["model_type"])
            grid_result.insert(2, "reg_strength", combo["reg_strength"])
            all_rows.append(grid_result)

    results = pd.concat(all_rows, ignore_index=True)

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_pickle(results_file)

    return results
