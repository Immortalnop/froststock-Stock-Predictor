"""Calibrated uncertainty: rolling split-conformal intervals and direction probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def rolling_conformal(
    pred: pd.Series, y: pd.Series, alpha: float = 0.2, calib_days: int = 126, horizon: int = 5,
    min_calib: int = 200,
) -> pd.DataFrame:
    """Split-conformal intervals pred +/- q, q from recent absolute residuals.

    For prediction date d only residuals of dates s <= d - (horizon+1) are used, i.e. those whose
    labels were already realised, so intervals are valid in live use. Returns lo, hi, covered.
    """
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    res = (df["y"] - df["p"]).abs()
    dates = pd.DatetimeIndex(sorted(df.index.get_level_values("date").unique()))
    by_date = {d: g.to_numpy() for d, g in res.groupby(level="date")}
    out_lo, out_hi = pd.Series(np.nan, index=pred.index), pd.Series(np.nan, index=pred.index)
    for i, d in enumerate(dates):
        end = i - (horizon + 1)
        if end < 0:
            continue
        window = dates[max(0, end - calib_days + 1) : end + 1]
        r = np.concatenate([by_date[w] for w in window])
        if len(r) < min_calib:
            continue
        k = min(len(r), int(np.ceil((len(r) + 1) * (1 - alpha))))
        q = np.sort(r)[k - 1]
        rows = pred.index[pred.index.get_level_values("date") == d]
        out_lo.loc[rows] = pred.loc[rows] - q
        out_hi.loc[rows] = pred.loc[rows] + q
    res_df = pd.DataFrame({"lo": out_lo, "hi": out_hi})
    yy = y.reindex(res_df.index)
    res_df["covered"] = ((yy >= res_df.lo) & (yy <= res_df.hi)).where(res_df.lo.notna() & yy.notna())
    return res_df


class DirectionCalibrator:
    """Isotonic map from a raw score (e.g. model prediction) to P(excess return > 0).
    Fit on a calibration block strictly before the period it is applied to."""

    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")

    def fit(self, score, went_up) -> "DirectionCalibrator":
        s, u = np.asarray(score, float), np.asarray(went_up, float)
        m = np.isfinite(s) & np.isfinite(u)
        self.iso.fit(s[m], u[m])
        return self

    def predict(self, score) -> np.ndarray:
        return self.iso.predict(np.asarray(score, float))


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2))


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)}).dropna()
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True)
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_up": g["y"].mean(), "n": g.size()})


def pinball(y, q_pred, tau: float) -> float:
    y, q = np.asarray(y, float), np.asarray(q_pred, float)
    m = np.isfinite(y) & np.isfinite(q)
    d = y[m] - q[m]
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
