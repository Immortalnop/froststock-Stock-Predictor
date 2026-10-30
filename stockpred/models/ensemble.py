"""Out-of-fold predictions and a walk-forward stacked ensemble (non-negative linear blend)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from stockpred.models.evaluate import rank_ic


def oof_predictions(
    factories: dict[str, Callable[[], object]],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit every model on each fold's train set, predict its test set.

    Returns (OOF frame with one column per model, fold id per row; -1 where not predicted).
    """
    oof = pd.DataFrame(np.nan, index=X.index, columns=list(factories))
    fold_id = np.full(len(X), -1)
    for k, (tr, te) in enumerate(folds):
        for name, make in factories.items():
            m = make().fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te, oof.columns.get_loc(name)] = m.predict(X.iloc[te])
        fold_id[te] = k
    return oof, fold_id


def _fold_ic(preds: pd.Series, y: pd.Series) -> float:
    ic = rank_ic(preds, y)
    return float(ic.mean()) if len(ic) else 0.0


def stack_walkforward(oof: pd.DataFrame, y: pd.Series, fold_id: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """Honest stacking evaluation: blend weights for fold k are fit only on folds < k.

    Fold 0 (no history) uses the equal-weight average. Returns (stacked prediction,
    weights table per fold).
    """
    out = pd.Series(np.nan, index=oof.index, name="stack")
    names = list(oof.columns)
    rows = []
    for k in sorted(set(fold_id[fold_id >= 0])):
        te = fold_id == k
        if k == 0:
            w = np.full(len(names), 1.0 / len(names))
        else:
            prev = (fold_id >= 0) & (fold_id < k)
            A, b = oof.loc[prev, names], y[prev]
            m = A.notna().all(axis=1) & b.notna()
            if m.sum() < 500:
                w = np.full(len(names), 1.0 / len(names))
            else:
                lr = LinearRegression(positive=True, fit_intercept=False).fit(A[m], b[m])
                w = lr.coef_
                w = w / w.sum() if w.sum() > 1e-12 else np.full(len(names), 1.0 / len(names))
        out[te] = oof.loc[te, names].to_numpy() @ w
        rows.append({"fold": k, **dict(zip(names, w))})
    return out, pd.DataFrame(rows)
