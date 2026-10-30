"""Optuna tuning of LightGBM with the objective computed on *inner* purged walk-forward folds
of the training data only (the outer test folds are never seen)."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from stockpred.models.cv import PurgedWalkForward
from stockpred.models.evaluate import rank_ic
from stockpred.models.gbm import GBM


def tune_gbm(
    X: pd.DataFrame, y: pd.Series, y_eval: pd.Series, horizon: int = 5, embargo: int = 2,
    n_trials: int = 25, inner_splits: int = 3, test_days: int = 63, min_train_days: int = 378,
    seed: int = 7,
) -> tuple[dict, float, int]:
    """Returns (best_params, best_inner_mean_IC, n_trials_run)."""
    inner = PurgedWalkForward(inner_splits, test_days, min_train_days, horizon, embargo)
    folds = list(inner.split(X.index))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 31),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 800, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        )
        ics = []
        for tr, te in folds:
            m = GBM(params=params, val_frac=0.0).fit(X.iloc[tr], y.iloc[tr])
            p = pd.Series(m.predict(X.iloc[te]), index=X.index[te])
            ic = rank_ic(p, y_eval.iloc[te])
            ics.append(ic.mean() if len(ic) else 0.0)
        return float(np.mean(ics))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value), n_trials
