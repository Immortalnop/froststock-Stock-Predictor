"""LightGBM wrappers: point forecast, quantiles, direction classifier. SHAP via native pred_contrib."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
    random_state=7, n_jobs=-1, verbose=-1,
)


def _time_val_split(X: pd.DataFrame, frac: float) -> np.ndarray:
    """Boolean mask: last `frac` of dates held out (time-ordered, with a 10-day gap dropped)."""
    d = X.index.get_level_values("date")
    ud = np.sort(d.unique())
    cut = ud[int(len(ud) * (1 - frac))]
    return np.asarray(d >= cut)


class GBM:
    def __init__(self, objective: str = "regression", params: dict | None = None,
                 val_frac: float = 0.15, early_stopping: int = 50, alpha: float | None = None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.objective, self.val_frac, self.early = objective, val_frac, early_stopping
        if objective == "quantile":
            p["alpha"] = alpha
        cls = lgb.LGBMClassifier if objective == "binary" else lgb.LGBMRegressor
        self.model = cls(objective=objective, **p)
        self.columns_: list[str] | None = None
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBM":
        m = y.notna().to_numpy()
        X, y = X[m], y[m]
        self.columns_ = list(X.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.val_frac and len(X) > 2000:
                v = _time_val_split(X, self.val_frac)
                # purge a margin so validation labels do not overlap training labels
                d = X.index.get_level_values("date")
                first_val = d[v].min()
                tr = ~v & (d < first_val - pd.Timedelta(days=14))
                self.model.fit(
                    X[tr], y[tr], eval_set=[(X[v], y[v])],
                    callbacks=[lgb.early_stopping(self.early, verbose=False)],
                )
                self.best_iteration_ = self.model.best_iteration_
            else:
                self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns_]
        if self.objective == "binary":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP values (TreeSHAP built into LightGBM); last column is the bias."""
        c = self.model.booster_.predict(X[self.columns_], pred_contrib=True)
        return pd.DataFrame(c, index=X.index, columns=self.columns_ + ["_bias"])

    def importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.columns_).sort_values(ascending=False)
