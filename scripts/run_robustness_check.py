import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import (
    CostModel,
    build_nudge_plan,
    compute_robustness_verdicts,
    evaluate_parameter_combination,
    expensive_combos_needed,
    raw_threshold_signal,
    run_expensive_combo,
    zscore_threshold_signal,
)

DEFAULT_DATA_PATH = "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_RESULTS_PATH = PROJECT_ROOT / "results" / "full_grid_results.pkl"
CACHE_DIR = PROJECT_ROOT / "results" / "backtest_cache"
ROBUSTNESS_PATH = PROJECT_ROOT / "results" / "robustness_check.pkl"

N_CANDIDATES = 15
MAX_LAG = 5
MIN_TRAIN_SIZE = 750
REFIT_FREQ = 21


def _predictions_cache_path(lookback, model_type, reg_strength) -> Path:
    return CACHE_DIR / f"predictions_L{lookback}_{model_type}_reg{reg_strength}.csv"


def evaluate_nudge_row(row: dict) -> float:
    cache_path = _predictions_cache_path(row["lookback"], row["model_type"], row["reg_strength"])
    predictions = pd.read_csv(cache_path, index_col="date", parse_dates=True)
    oos = predictions.dropna(subset=["y_pred"])
    y_pred, y_true = oos["y_pred"], oos["y_true"]

    if row["method"] == "raw":
        signal = raw_threshold_signal(y_pred, row["threshold"])
    else:
        signal = zscore_threshold_signal(y_pred, int(row["lookback"]), row["threshold"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate_parameter_combination(
            y_pred,
            y_true,
            signal,
            hold_days=int(row["hold_days"]),
            overlap_rule=row["overlap_rule"],
            cost_model=CostModel(spread_bps=row["spread_bps"]),
        )
    return result["net_sharpe"], result["net_total_return"]


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    force_rerun = "--force" in sys.argv

    if ROBUSTNESS_PATH.exists() and not force_rerun:
        print(f"Loading cached robustness check from {ROBUSTNESS_PATH}")
        results = pd.read_pickle(ROBUSTNESS_PATH)
    else:
        grid_results = pd.read_pickle(GRID_RESULTS_PATH)
        eligible = grid_results[grid_results["meets_min_frequency"]].copy()
        candidates = eligible.sort_values("net_sharpe", ascending=False).head(N_CANDIDATES).reset_index(drop=True)

        print(f"Top {N_CANDIDATES} candidates:")
        print(candidates[["lookback", "model_type", "reg_strength", "method", "threshold", "hold_days", "overlap_rule", "spread_bps", "net_sharpe"]].to_string())
        print()

        plan = build_nudge_plan(candidates)
        needed = expensive_combos_needed(plan)
        missing = [c for c in needed if not _predictions_cache_path(*c).exists()]
        print(f"Nudge plan: {len(plan)} rows. Expensive combos referenced: {len(needed)}, missing from cache: {len(missing)}")

        if missing:
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        run_expensive_combo,
                        data_path,
                        lookback,
                        model_type,
                        reg_strength,
                        MAX_LAG,
                        MIN_TRAIN_SIZE,
                        REFIT_FREQ,
                        str(CACHE_DIR),
                    ): (lookback, model_type, reg_strength)
                    for lookback, model_type, reg_strength in missing
                }
                done = 0
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    print(f"  [{done}/{len(missing)}] done: {futures[future]}")

        print("\nEvaluating nudge plan rows...")
        net_sharpes, net_returns = [], []
        for _, row in plan.iterrows():
            if not row["valid"]:
                net_sharpes.append(float("nan"))
                net_returns.append(float("nan"))
                continue
            sharpe, total_return = evaluate_nudge_row(row)
            net_sharpes.append(sharpe)
            net_returns.append(total_return)
        plan["net_sharpe"] = net_sharpes
        plan["net_total_return"] = net_returns

        verdicts = compute_robustness_verdicts(plan)

        candidate_labels = candidates.apply(
            lambda r: f"L{r['lookback']} {r['model_type']} {r['method']}={r['threshold']} hold={r['hold_days']} {r['overlap_rule']} {r['spread_bps']}bps",
            axis=1,
        )
        plan["candidate_label"] = plan["candidate_id"].map(candidate_labels)
        verdicts["candidate_label"] = verdicts["candidate_id"].map(candidate_labels)

        results = {"plan": plan, "verdicts": verdicts, "candidates": candidates}
        ROBUSTNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(results, ROBUSTNESS_PATH)

    plan, verdicts, candidates = results["plan"], results["verdicts"], results["candidates"]

    pd.set_option("display.width", 200)
    print("\n=== ROBUSTNESS VERDICTS ===")
    print(verdicts[["candidate_id", "candidate_label", "baseline_net_sharpe", "verdict",
                     "worst_one_step_relative_drop", "worst_one_step_parameter", "worst_one_step_direction"]].to_string(index=False))

    print(f"\nRobust: {(verdicts['verdict']=='robust').sum()}   Fragile: {(verdicts['verdict']=='fragile').sum()}")
