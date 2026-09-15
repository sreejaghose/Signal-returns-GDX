import os
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from signal_gdx import build_model_for_grid, run_expensive_combo, run_full_grid

DEFAULT_DATA_PATH = os.environ.get(
    "GDX_DATA_PATH",
    "/root/.claude/uploads/c63a269f-18b7-5367-837e-b3b8d1b00c7c/1f2fe993-Sreeja_-_ETF_data_-_IN-SAMPLE.xlsx",
)

TMP_DIR = Path("/tmp/claude-0/-home-user-Signal-returns-GDX/c63a269f-18b7-5367-837e-b3b8d1b00c7c/scratchpad/grid_runner_test")


@pytest.fixture(autouse=True)
def clean_tmp_dir():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    yield
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


def test_build_model_for_grid_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        build_model_for_grid("not_a_model", 1.0)


def test_build_model_for_grid_returns_distinct_estimators_per_type():
    en = build_model_for_grid("elasticnet", 0.01)
    lgbm = build_model_for_grid("lightgbm", 1.0)
    assert en.__class__.__name__ == "Pipeline"
    assert lgbm.__class__.__name__ == "LGBMRegressor"


def test_run_expensive_combo_caches_to_disk_and_skips_on_rerun():
    cache_dir = TMP_DIR / "cache"
    result1 = run_expensive_combo(
        data_path=DEFAULT_DATA_PATH,
        lookback=20,
        model_type="lightgbm",
        reg_strength=1.0,
        max_lag=5,
        min_train_size=700,
        refit_freq=250,
        cache_dir=str(cache_dir),
    )
    assert result1["was_cached"] is False
    cache_path = Path(result1["cache_path"])
    assert cache_path.exists()

    mtime_before = cache_path.stat().st_mtime

    result2 = run_expensive_combo(
        data_path=DEFAULT_DATA_PATH,
        lookback=20,
        model_type="lightgbm",
        reg_strength=1.0,
        max_lag=5,
        min_train_size=700,
        refit_freq=250,
        cache_dir=str(cache_dir),
    )
    assert result2["was_cached"] is True
    assert cache_path.stat().st_mtime == mtime_before  # file was never rewritten

    predictions = pd.read_csv(cache_path, index_col="date", parse_dates=True)
    assert {"y_true", "y_pred", "fold"} <= set(predictions.columns)


def test_run_full_grid_produces_one_row_per_full_combination():
    results_path = TMP_DIR / "results.pkl"
    cache_dir = TMP_DIR / "cache"

    results = run_full_grid(
        data_path=DEFAULT_DATA_PATH,
        lookback_grid=[20],
        model_types=["lightgbm"],
        reg_strength_grid={"lightgbm": [1.0]},
        raw_thresholds=[0.001],
        zscore_thresholds=[1.0],
        hold_days_grid=[1, 3],
        overlap_rules=["ignore_new_signal"],
        cost_bps_grid=[2, 5],
        results_path=str(results_path),
        cache_dir=str(cache_dir),
        min_train_size=700,
        refit_freq=250,
        max_workers=2,
    )

    # 1 lookback x 1 model x 1 reg_strength x 2 signal configs (1 raw + 1 zscore)
    # x 2 hold_days x 1 overlap_rule x 2 cost_bps = 8 rows
    assert len(results) == 8
    required_cols = {
        "lookback", "model_type", "reg_strength", "method", "threshold",
        "hold_days", "overlap_rule", "spread_bps",
        "trades_per_month", "meets_min_frequency",
        "gross_sharpe", "net_sharpe", "hold_day_decay_gross", "hold_day_decay_net",
    }
    assert required_cols <= set(results.columns)
    assert results_path.exists()


def test_run_full_grid_second_call_loads_from_disk_without_recomputing():
    results_path = TMP_DIR / "results.pkl"
    cache_dir = TMP_DIR / "cache"

    kwargs = dict(
        data_path=DEFAULT_DATA_PATH,
        lookback_grid=[20],
        model_types=["lightgbm"],
        reg_strength_grid={"lightgbm": [1.0]},
        raw_thresholds=[0.001],
        zscore_thresholds=[],
        hold_days_grid=[1],
        overlap_rules=["ignore_new_signal"],
        cost_bps_grid=[5],
        results_path=str(results_path),
        cache_dir=str(cache_dir),
        min_train_size=700,
        refit_freq=250,
        max_workers=2,
    )

    first = run_full_grid(**kwargs)
    assert results_path.exists()

    t0 = time.time()
    second = run_full_grid(**kwargs)
    elapsed = time.time() - t0

    pd.testing.assert_frame_equal(first, second)
    assert elapsed < 2.0  # loaded from disk, did not rerun any backtest


def test_run_full_grid_force_rerun_recomputes_even_if_file_exists():
    results_path = TMP_DIR / "results.pkl"
    cache_dir = TMP_DIR / "cache"

    kwargs = dict(
        data_path=DEFAULT_DATA_PATH,
        lookback_grid=[20],
        model_types=["lightgbm"],
        reg_strength_grid={"lightgbm": [1.0]},
        raw_thresholds=[0.001],
        zscore_thresholds=[],
        hold_days_grid=[1],
        overlap_rules=["ignore_new_signal"],
        cost_bps_grid=[5],
        results_path=str(results_path),
        cache_dir=str(cache_dir),
        min_train_size=700,
        refit_freq=250,
        max_workers=2,
    )

    run_full_grid(**kwargs)
    mtime_before = results_path.stat().st_mtime
    time.sleep(1.1)  # ensure a distinguishable mtime if rewritten
    run_full_grid(**kwargs, force_rerun=True)
    assert results_path.stat().st_mtime > mtime_before
