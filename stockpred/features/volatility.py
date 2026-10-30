"""Volatility estimators and a leak-free rolling GARCH forecast."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockpred.features.technical import adjust_ohlc

log = logging.getLogger(__name__)
ANN = np.sqrt(252.0)


def garch_vol_forecast(
    ret: pd.Series, refit_every: int = 63, min_obs: int = 504, o: int = 0
) -> pd.Series:
    """Annualised next-day vol forecast. out[t] uses returns through t only.

    Parameters are re-estimated every `refit_every` days on data up to that point; between
    refits the variance recursion is applied with the *fixed* parameters (no refit leakage).
    o=1 gives GJR-GARCH (leverage effect), the common stand-in for EGARCH asymmetry.
    """
    from arch import arch_model

    r = (ret.dropna() * 100.0).astype(float)
    n = len(r)
    out = pd.Series(np.nan, index=r.index, name="garch_vol")
    if n <= min_obs:
        return out.reindex(ret.index)
    vals = r.to_numpy()
    s = min_obs
    while s < n:
        end = min(s + refit_every, n)  # forecasts for t in [s-1, end-2] (+ final)
        try:
            res = arch_model(vals[:s], mean="Constant", vol="GARCH", p=1, o=o, q=1,
                             dist="normal", rescale=False).fit(disp="off", show_warning=False)
            p = res.params
            mu, om, al = p["mu"], p["omega"], p["alpha[1]"]
            ga = p["gamma[1]"] if o else 0.0
            be = p["beta[1]"]
            s2 = float(res.conditional_volatility[-1] ** 2)  # variance of r[s-1] given <= s-2
            for t in range(s - 1, end):
                e = vals[t] - mu
                s2 = om + al * e * e + ga * e * e * (e < 0) + be * s2  # variance for t+1
                out.iloc[t] = np.sqrt(s2) / 100.0 * ANN
        except Exception as exc:
            log.warning("GARCH fit failed at %d: %s", s, exc)
        s = end
    return out.reindex(ret.index)


def volatility_features(px: pd.DataFrame, garch: bool = True) -> pd.DataFrame:
    a = adjust_ohlc(px)
    c = a["close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=a.index)
    f["rv_20"] = lr.rolling(20).std() * ANN
    f["rv_60"] = lr.rolling(60).std() * ANN
    hl = np.log(a["high"] / a["low"])
    f["parkinson_20"] = np.sqrt((hl**2).rolling(20).mean() / (4 * np.log(2))) * ANN
    co = np.log(a["close"] / a["open"])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    f["gk_20"] = np.sqrt(gk.rolling(20).mean().clip(lower=0)) * ANN
    f["vol_of_vol"] = f["rv_20"].rolling(60).std() / f["rv_20"].rolling(60).mean()
    f["rv_ratio"] = f["rv_20"] / f["rv_60"]
    if garch:
        g = garch_vol_forecast(lr)
        f["garch_vol"] = g
        f["garch_ratio"] = g / f["rv_20"]
    return f.replace([np.inf, -np.inf], np.nan)
