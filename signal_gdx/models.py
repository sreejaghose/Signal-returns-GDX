import lightgbm as lgb
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def make_elasticnet_model(
    l1_ratios: list[float] | None = None,
    n_alphas: int = 50,
    inner_cv_splits: int = 5,
    max_iter: int = 20000,
    tol: float = 1e-3,
    random_state: int = 42,
) -> Pipeline:
    """ElasticNet with an internal alpha/l1_ratio sweep, refit fresh each fold.

    ElasticNetCV sweeps l1_ratio over `l1_ratios` and alpha over an
    automatically generated `n_alphas`-point path for each l1_ratio, picking
    the combination with the best inner cross-validated score. The inner CV
    is a TimeSeriesSplit (not a shuffled KFold) so the hyperparameter search
    itself never lets later rows of the fold's training window leak into
    earlier ones. Wrapped in a Pipeline with StandardScaler since ElasticNet
    is scale-sensitive; the whole pipeline is a single sklearn estimator, so
    it clones and fits/predicts through walk_forward_backtest unchanged.
    """
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=l1_ratios,
                    alphas=n_alphas,
                    cv=TimeSeriesSplit(n_splits=inner_cv_splits),
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_shallow_lightgbm(random_state: int = 42, **overrides) -> lgb.LGBMRegressor:
    """A deliberately shallow, heavily regularized LightGBM regressor.

    Daily return features are mostly noise, so this favors strong
    regularization (few, shallow trees, high min_child_samples, L1/L2
    penalties, row/column subsampling) over raw fitting power.
    """
    params = dict(
        n_estimators=50,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )
    params.update(overrides)
    return lgb.LGBMRegressor(**params)
