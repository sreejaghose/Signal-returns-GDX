import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gdx import (
    CostModel,
    compute_holdout_start,
    evaluate_candidate_on_holdout,
    flag_holdout_degradation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_RESULTS_PATH = PROJECT_ROOT / "results" / "full_grid_results.pkl"
ROBUSTNESS_PATH = PROJECT_ROOT / "results" / "robustness_check.pkl"
CACHE_DIR = PROJECT_ROOT / "results" / "backtest_cache"
HOLDOUT_RESULTS_PATH = PROJECT_ROOT / "results" / "holdout_check.pkl"

HOLDOUT_FRACTION = 0.20
N_CANDIDATES = 15


def _predictions_cache_path(lookback, model_type, reg_strength) -> Path:
    return CACHE_DIR / f"predictions_L{lookback}_{model_type}_reg{reg_strength}.csv"


if __name__ == "__main__":
    force_rerun = "--force" in sys.argv

    robustness = pd.read_pickle(ROBUSTNESS_PATH)
    verdicts, candidates = robustness["verdicts"], robustness["candidates"]

    robust_ids = verdicts[verdicts["verdict"] == "robust"]["candidate_id"].tolist()
    if robust_ids:
        chosen = candidates.iloc[robust_ids].reset_index(drop=True)
        print(f"Using {len(chosen)} candidates flagged 'robust' in the step-12 check.")
    else:
        chosen = candidates.head(N_CANDIDATES).reset_index(drop=True)
        print(
            "No candidates were flagged 'robust' in the step-12 local robustness check "
            f"(all {len(candidates)} came back 'fragile'). Falling back to the raw top-"
            f"{N_CANDIDATES}-by-net-Sharpe list instead, per the fallback rule -- "
            "this holdout check will very likely confirm the fragility finding rather "
            "than contradict it."
        )

    # any one candidate's predictions file has the full underlying date index
    reference_index = pd.read_csv(
        _predictions_cache_path(*chosen.iloc[0][["lookback", "model_type", "reg_strength"]]),
        index_col="date", parse_dates=True,
    ).index
    holdout_start = compute_holdout_start(reference_index, HOLDOUT_FRACTION)
    print(f"Holdout: last {HOLDOUT_FRACTION:.0%} of the full {reference_index.min().date()}..{reference_index.max().date()} "
          f"span -> holdout starts {holdout_start.date()}\n")

    rows = []
    for _, cand in chosen.iterrows():
        cache_path = _predictions_cache_path(cand["lookback"], cand["model_type"], cand["reg_strength"])
        predictions = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        oos = predictions.dropna(subset=["y_pred"])
        y_pred, y_true = oos["y_pred"], oos["y_true"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holdout_result = evaluate_candidate_on_holdout(
                y_pred, y_true,
                method=cand["method"], threshold=cand["threshold"], lookback=int(cand["lookback"]),
                hold_days=int(cand["hold_days"]), overlap_rule=cand["overlap_rule"],
                cost_model=CostModel(spread_bps=cand["spread_bps"]),
                holdout_start=holdout_start,
            )

        degradation = flag_holdout_degradation(cand["net_sharpe"], holdout_result["net_sharpe"])

        rows.append({
            "lookback": cand["lookback"], "model_type": cand["model_type"], "reg_strength": cand["reg_strength"],
            "method": cand["method"], "threshold": cand["threshold"], "hold_days": cand["hold_days"],
            "overlap_rule": cand["overlap_rule"], "spread_bps": cand["spread_bps"],
            "in_sample_net_sharpe": cand["net_sharpe"],
            "in_sample_net_total_return": cand["net_total_return"],
            "in_sample_net_hit_rate": cand["net_hit_rate"],
            "in_sample_net_max_drawdown": cand["net_max_drawdown"],
            "in_sample_trades_per_month": cand["trades_per_month"],
            "holdout_net_sharpe": holdout_result["net_sharpe"],
            "holdout_net_total_return": holdout_result["net_total_return"],
            "holdout_net_hit_rate": holdout_result["net_hit_rate"],
            "holdout_net_max_drawdown": holdout_result["net_max_drawdown"],
            "holdout_trades_per_month": holdout_result["trades_per_month"],
            "holdout_n_trades": holdout_result["n_trades"],
            "sharpe_relative_drop": degradation["relative_drop"],
            "flagged_overfit": degradation["flagged_overfit"],
        })

    results = pd.DataFrame(rows)
    HOLDOUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_pickle(HOLDOUT_RESULTS_PATH)

    pd.set_option("display.width", 220)
    cols = ["lookback", "model_type", "method", "threshold", "hold_days", "overlap_rule", "spread_bps",
            "in_sample_net_sharpe", "holdout_net_sharpe", "sharpe_relative_drop", "flagged_overfit",
            "holdout_n_trades", "holdout_trades_per_month"]
    print(results[cols].to_string(index=True))
    print(f"\nFlagged as likely overfit despite robustness check: {results['flagged_overfit'].sum()} of {len(results)}")
