"""Portfolio risk: historical / parametric / Monte Carlo VaR and CVaR, stress tests, limit checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

STRESS_WINDOWS = {
    "gfc_lehman": ("2008-09-12", "2008-10-10"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "rates_2022h1": ("2022-01-03", "2022-06-16"),
}


def hist_var_cvar(r: pd.Series, level: float = 0.95) -> dict[str, float]:
    """Loss quantile and expected shortfall, both reported as positive fractions of NAV."""
    x = r.dropna().to_numpy()
    if len(x) < 20:
        return {"var": np.nan, "cvar": np.nan}
    q = np.quantile(x, 1 - level)
    tail = x[x <= q]
    return {"var": float(-q), "cvar": float(-tail.mean())}


def parametric_var(r: pd.Series, level: float = 0.95) -> float:
    x = r.dropna()
    return float(-(x.mean() + stats.norm.ppf(1 - level) * x.std()))


def monte_carlo_var_cvar(w: pd.Series, returns: pd.DataFrame, level: float = 0.95, horizon: int = 1,
                         n_sims: int = 20000, df: int = 5, seed: int = 0) -> dict[str, float]:
    """Multivariate-t simulation with the sample covariance (fat tails, correlation preserved)."""
    R = returns.dropna(how="any")
    names = [c for c in w.index if c in R.columns]
    R, ww = R[names], w[names].to_numpy()
    mu, cov = R.mean().to_numpy(), R.cov().to_numpy()
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov * (df - 2) / df + 1e-12 * np.eye(len(names)))
    z = rng.standard_normal((n_sims, horizon, len(names))) @ L.T
    chi = rng.chisquare(df, size=(n_sims, horizon, 1)) / df
    sims = mu + z / np.sqrt(chi)
    pnl = (sims @ ww).sum(axis=1)
    q = np.quantile(pnl, 1 - level)
    return {"var": float(-q), "cvar": float(-pnl[pnl <= q].mean())}


def stress_replay(w: pd.Series, returns: pd.DataFrame, windows: dict | None = None) -> pd.DataFrame:
    """Replay current weights through historical crisis windows (compounded asset returns)."""
    out = {}
    for name, (a, b) in (windows or STRESS_WINDOWS).items():
        seg = returns.loc[a:b, [c for c in w.index if c in returns.columns]]
        if len(seg) < 5:
            continue
        cum = (1 + seg.fillna(0)).prod() - 1
        out[name] = {"pnl": float((cum * w[cum.index]).sum()), "days": len(seg)}
    return pd.DataFrame(out).T


def stress_shock(w: pd.Series, betas: pd.Series, market_shock: float = -0.10) -> float:
    """Hypothetical instantaneous market move: pnl = shock * sum(w_i * beta_i)."""
    return float(market_shock * (w * betas.reindex(w.index).fillna(1.0)).sum())


@dataclass
class RiskLimits:
    max_gross: float = 1.0
    max_net: float = 0.05
    max_name: float = 0.10
    max_sector_net: float = 0.10
    max_beta: float = 0.10
    max_var95_1d: float = 0.02  # 1-day 95% historical VaR as fraction of NAV
    stop_drawdown: float = 0.15


@dataclass
class LimitReport:
    ok: bool
    violations: list[str] = field(default_factory=list)


def check_limits(w: pd.Series, lim: RiskLimits, betas: pd.Series | None = None, sectors: pd.Series | None = None,
                 var95: float | None = None, drawdown: float | None = None) -> LimitReport:
    v = []
    if w.abs().sum() > lim.max_gross + 1e-9:
        v.append(f"gross {w.abs().sum():.3f} > {lim.max_gross}")
    if abs(w.sum()) > lim.max_net + 1e-9:
        v.append(f"net {w.sum():+.3f} beyond +/-{lim.max_net}")
    if w.abs().max() > lim.max_name + 1e-9:
        v.append(f"name {w.abs().idxmax()} {w.abs().max():.3f} > {lim.max_name}")
    if betas is not None:
        b = float((w * betas.reindex(w.index).fillna(1.0)).sum())
        if abs(b) > lim.max_beta + 1e-9:
            v.append(f"beta {b:+.3f} beyond +/-{lim.max_beta}")
    if sectors is not None:
        net = w.groupby(sectors.reindex(w.index).fillna("?")).sum()
        for s, x in net.items():
            if abs(x) > lim.max_sector_net + 1e-9:
                v.append(f"sector {s} net {x:+.3f} beyond +/-{lim.max_sector_net}")
    if var95 is not None and var95 > lim.max_var95_1d:
        v.append(f"VaR95 {var95:.3%} > {lim.max_var95_1d:.2%}")
    if drawdown is not None and drawdown >= lim.stop_drawdown:
        v.append(f"drawdown {drawdown:.1%} >= stop {lim.stop_drawdown:.0%}")
    return LimitReport(not v, v)


def scale_to_limits(w: pd.Series, lim: RiskLimits, var95: float | None = None) -> pd.Series:
    """Clip names, then scale down uniformly to satisfy gross and VaR limits. Net/beta are the optimiser's job."""
    w = w.clip(-lim.max_name, lim.max_name)
    s = 1.0
    if w.abs().sum() > lim.max_gross:
        s = min(s, lim.max_gross / w.abs().sum())
    if var95 is not None and var95 > lim.max_var95_1d:
        s = min(s, lim.max_var95_1d / var95)
    return w * s


def risk_report(w: pd.Series, returns: pd.DataFrame, betas: pd.Series | None = None) -> dict:
    cols = [c for c in w.index if c in returns.columns]
    hist = (returns[cols].fillna(0) * w[cols]).sum(axis=1)
    out = {"hist_1d_95": hist_var_cvar(hist, 0.95), "hist_1d_99": hist_var_cvar(hist, 0.99),
           "parametric_1d_95": parametric_var(hist, 0.95),
           "mc_5d_95": monte_carlo_var_cvar(w[cols], returns[cols], 0.95, horizon=5),
           "stress": stress_replay(w[cols], returns[cols]).to_dict("index")}
    if betas is not None:
        out["shock_-10%"] = stress_shock(w, betas, -0.10)
    return out
