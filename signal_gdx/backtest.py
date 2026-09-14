from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp | None
    embargo_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def walk_forward_backtest(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    estimator,
    min_train_size: int,
    refit_freq: int,
    embargo: int,
    window: str = "expanding",
    train_window_size: int | None = None,
) -> tuple[pd.DataFrame, list[Fold]]:
    """Walk-forward, one-day-ahead backtest with an embargo gap between train and test.

    Refits `estimator` (cloned fresh each time via sklearn.base.clone, so any
    sklearn-style estimator works) every `refit_freq` trading days. Each
    refit produces predictions for the next block of `refit_freq` days, one
    day at a time -- each prediction uses only that day's own feature row,
    so every prediction is a genuine one-step-ahead forecast; refit_freq only
    controls how often the model is retrained, not how far ahead it predicts.

    Between the last training day and the first test day of each fold, an
    embargo of `embargo` days is skipped entirely (used for neither train nor
    test) to keep rolling-window features from bleeding across the boundary.
    Those embargoed days re-enter the training set on the *next* fold, once
    they are safely in the past.

    Args:
        feature_df: full feature DataFrame indexed by date (as produced by
            build_features). Rows with any NaN in feature_cols or target_col
            are dropped before splitting.
        feature_cols: column names to use as model inputs.
        target_col: column name of the (already-shifted) target.
        estimator: an unfitted sklearn-style estimator (must support
            sklearn.base.clone, .fit(X, y), .predict(X)).
        min_train_size: number of valid rows in the first training window.
        refit_freq: number of days predicted per fold before refitting.
        embargo: number of valid rows skipped between train and test in each fold.
        window: "expanding" (train always starts at row 0) or "rolling"
            (train is the most recent train_window_size rows).
        train_window_size: required when window="rolling"; ignored otherwise.

    Returns:
        (predictions, folds) where predictions is a DataFrame indexed on
        feature_df's full index with columns y_true, y_pred, fold (NaN for
        the initial burn-in period with no out-of-fold prediction yet), and
        folds is a list of Fold records describing each split's exact date
        boundaries.
    """
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be 'expanding' or 'rolling'")
    if window == "rolling" and not train_window_size:
        raise ValueError("train_window_size is required when window='rolling'")
    if min_train_size < 1 or refit_freq < 1 or embargo < 0:
        raise ValueError("min_train_size and refit_freq must be >= 1, embargo must be >= 0")

    valid = feature_df.dropna(subset=feature_cols + [target_col])
    dates = valid.index
    X = valid[feature_cols].to_numpy()
    y = valid[target_col].to_numpy()
    n = len(valid)

    if min_train_size + embargo >= n:
        raise ValueError(
            f"min_train_size ({min_train_size}) + embargo ({embargo}) leaves no room "
            f"for test data out of {n} valid rows"
        )

    predictions = pd.DataFrame(
        {"y_true": np.nan, "y_pred": np.nan, "fold": np.nan},
        index=feature_df.index,
    )
    predictions.loc[dates, "y_true"] = y

    folds: list[Fold] = []
    train_end = min_train_size
    fold_id = 0

    while True:
        test_start = train_end + embargo
        if test_start >= n:
            break
        test_end = min(test_start + refit_freq, n)

        train_start = 0 if window == "expanding" else max(0, train_end - train_window_size)
        train_pos = np.arange(train_start, train_end)
        test_pos = np.arange(test_start, test_end)

        model = clone(estimator)
        model.fit(X[train_pos], y[train_pos])
        preds = model.predict(X[test_pos])

        test_dates = dates[test_pos]
        predictions.loc[test_dates, "y_pred"] = preds
        predictions.loc[test_dates, "fold"] = fold_id

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=dates[train_start],
                train_end=dates[train_end - 1],
                embargo_start=dates[train_end] if embargo > 0 else None,
                embargo_end=dates[test_start - 1] if embargo > 0 else None,
                test_start=dates[test_start],
                test_end=dates[test_end - 1],
                n_train=len(train_pos),
                n_test=len(test_pos),
            )
        )

        train_end = test_end
        fold_id += 1

    return predictions, folds
