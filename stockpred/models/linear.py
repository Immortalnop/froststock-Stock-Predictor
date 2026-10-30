"""Regularised linear models with train-only preprocessing (no leakage via scaling/imputation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to train-set quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            self.lo_ = np.nanquantile(X, self.lower, axis=0)
            self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


class LinearModel:
    def __init__(self, kind: str = "ridge", alpha: float = 10.0, l1_ratio: float = 0.2):
        est = Ridge(alpha=alpha) if kind == "ridge" else ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        self.pipe = Pipeline([
            ("win", Winsorizer()),
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("est", est),
        ])
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        m = y.notna()
        self.columns_ = list(X.columns)
        self.pipe.fit(X[m].replace([np.inf, -np.inf], np.nan), y[m])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X[self.columns_].replace([np.inf, -np.inf], np.nan))

    def coefficients(self) -> pd.Series:
        return pd.Series(self.pipe.named_steps["est"].coef_, index=self.columns_).sort_values(
            key=np.abs, ascending=False)
