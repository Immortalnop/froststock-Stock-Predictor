"""Cross-sectional transforms, rolling beta and factor-residual (alpha) features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date percentile rank centred on 0 (range ~[-0.5, 0.5]); panel index = (date, symbol)."""
    r = panel[cols].groupby(level="date").rank(pct=True) - 0.5
    r.columns = [f"{c}_csr" for c in cols]
    return r


def sector_relative(panel: pd.DataFrame, cols: list[str], sector: pd.Series) -> pd.DataFrame:
    """value minus same-date sector mean. `sector` maps symbol -> sector."""
    sec = panel.index.get_level_values("symbol").map(sector).to_numpy()
    out = {}
    for c in cols:
        g = panel[c].groupby([panel.index.get_level_values("date"), sec])
        out[f"{c}_secrel"] = panel[c] - g.transform("mean")
    return pd.DataFrame(out, index=panel.index)


def rolling_beta(ret: pd.Series, mkt: pd.Series, window: int = 60) -> pd.Series:
    """Rolling CAPM beta using data through t (cov/var)."""
    cov = ret.rolling(window, min_periods=window // 2).cov(mkt)
    var = mkt.rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0, np.nan)).rename("beta_mkt")


def factor_exposures(
    ret: pd.Series, factors: pd.DataFrame, window: int = 126, rf: pd.Series | None = None,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Rolling OLS of excess returns on factors.

    Betas for day t are estimated on [t-window, t-1], so `resid_t = r_t - x_t' beta_{t-1}` is
    an out-of-sample alpha (no look-ahead into day t). Returns betas (as of t-1) and resid.
    """
    f = factors.reindex(ret.index)
    y = (ret - (rf.reindex(ret.index) if rf is not None else 0.0)).to_numpy(float)
    X = f.to_numpy(float)
    n, k = X.shape
    betas = np.full((n, k), np.nan)
    resid = np.full(n, np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xc = np.column_stack([np.ones(n), X])
    for t in range(window, n):
        lo = t - window
        m = ok[lo:t]
        if m.sum() < window * 0.8:
            continue
        A, b = Xc[lo:t][m], y[lo:t][m]
        coef = np.linalg.solve(A.T @ A + ridge * np.eye(k + 1), A.T @ b)
        betas[t] = coef[1:]
        if ok[t]:
            resid[t] = y[t] - Xc[t] @ coef
    out = pd.DataFrame(betas, index=ret.index, columns=[f"beta_{c}" for c in factors.columns])
    out["resid_ret"] = resid
    out["resid_mom_21"] = out["resid_ret"].rolling(21, min_periods=15).sum()
    return out
