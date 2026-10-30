"""Signal-quality metrics: rank IC, HAC t-stats, hit rate with CI, quantile spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(pred: pd.Series, target: pd.Series, min_names: int = 5) -> pd.Series:
    """Per-date Spearman correlation across symbols."""
    df = pd.DataFrame({"p": pred, "y": target}).dropna()
    g = df.groupby(level="date")
    pr = g["p"].rank()
    yr = g["y"].rank()
    d = pd.DataFrame({"p": pr, "y": yr})
    cnt = g["p"].transform("size")
    d = d[cnt >= min_names]
    with np.errstate(invalid="ignore", divide="ignore"):  # constant cross-sections give NaN, dropped
        out = d.groupby(level="date").apply(lambda x: x["p"].corr(x["y"]))
    return out.dropna().rename("rank_ic")


def hac_tstat(x: pd.Series, lags: int) -> float:
    """t-stat of the mean with Newey-West errors (overlapping forward returns autocorrelate)."""
    import statsmodels.api as sm

    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "ic_t_hac": np.nan,
                "ic_pos_frac": np.nan, "n_dates": 0}
    sd = float(ic.std())
    return {
        "ic_mean": float(ic.mean()), "ic_std": sd,
        "ic_ir": float(ic.mean() / sd) if sd > 0 else np.nan,
        "ic_t_hac": hac_tstat(ic, lags=max(1, horizon)),
        "ic_pos_frac": float((ic > 0).mean()), "n_dates": int(len(ic)),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def hit_rate(pred: pd.Series, target: pd.Series) -> dict[str, float]:
    """Directional accuracy of sign(pred) vs sign(target). The CI assumes independence, which
    overlapping horizons violate: treat it as optimistic."""
    df = pd.DataFrame({"p": pred, "y": target}).dropna()
    df = df[(df.p != 0) & (df.y != 0)]
    k, n = int((np.sign(df.p) == np.sign(df.y)).sum()), len(df)
    lo, hi = wilson(k, n)
    return {"hit_rate": k / n if n else np.nan, "hit_lo": lo, "hit_hi": hi, "n": n}


def quantile_spread(pred: pd.Series, target: pd.Series, q: int = 5) -> dict[str, float]:
    """Mean target of top vs bottom pred quantile per date (gross, before costs)."""
    df = pd.DataFrame({"p": pred, "y": target}).dropna()
    def one(g):
        if len(g) < q * 2:
            return pd.Series({"top": np.nan, "bot": np.nan})
        r = g["p"].rank(method="first")
        b = pd.qcut(r, q, labels=False)
        return pd.Series({"top": g.loc[b == q - 1, "y"].mean(), "bot": g.loc[b == 0, "y"].mean()})
    s = df.groupby(level="date").apply(one).dropna()
    return {"top_mean": float(s["top"].mean()), "bottom_mean": float(s["bot"].mean()),
            "spread": float((s["top"] - s["bot"]).mean()), "n_dates": int(len(s))}


def evaluate_signal(pred: pd.Series, target: pd.Series, horizon: int) -> dict[str, float]:
    ic = rank_ic(pred, target)
    out = ic_summary(ic, horizon)
    out.update(hit_rate(pred, target))
    out.update({f"q_{k}": v for k, v in quantile_spread(pred, target).items()})
    return out
