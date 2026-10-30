"""Covariance shrinkage and portfolio construction: inverse-vol, min-variance, HRP, Black-Litterman,
and a constrained long/short mean-variance optimiser with an L1 turnover penalty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrunk daily covariance. Rows with any NaN are dropped; pass a clean window."""
    r = returns.dropna(how="any")
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def signal_to_alpha(score: pd.Series, vol_daily: pd.Series, ic: float, horizon: int) -> pd.Series:
    """Grinold: alpha_i = IC * sigma_i(horizon) * z_i. Output is the expected excess return over the horizon."""
    z = (score - score.mean()) / (score.std(ddof=0) or 1.0)
    return ic * vol_daily * np.sqrt(horizon) * z


def evidence_scale(t_stat: float | None, full_at: float = 2.0) -> float:
    """Scale alpha by walk-forward evidence: 0 when the IC t-stat is not positive, 1 at t >= full_at."""
    if t_stat is None or not np.isfinite(t_stat):
        return 0.0
    return float(np.clip(t_stat / full_at, 0.0, 1.0))


def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    w = 1.0 / np.sqrt(np.diag(cov))
    return pd.Series(w / w.sum(), index=cov.index)


def min_variance(cov: pd.DataFrame, max_w: float = 1.0) -> pd.Series:
    n = len(cov)
    S = cov.to_numpy()
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w, method="SLSQP",
                   bounds=[(0, max_w)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=cov.index)


def _quasi_diag(link: np.ndarray) -> list[int]:
    return list(leaves_list(link))


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(cov: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado): cluster by correlation distance, recursive bisection."""
    S = cov.to_numpy()
    sd = np.sqrt(np.diag(S))
    corr = np.clip(S / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    order = _quasi_diag(linkage(squareform(dist, checks=False), method="single"))
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters if len(c) > 1 for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for k in range(0, len(clusters), 2):
            a, b = clusters[k], clusters[k + 1]
            va, vb = _cluster_var(S, a), _cluster_var(S, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return pd.Series(w.sort_index().to_numpy(), index=cov.index)


def black_litterman(cov: pd.DataFrame, mkt_w: pd.Series, view_alpha: pd.Series, *, risk_aversion: float = 2.5,
                    tau: float = 0.05, view_conf: float = 0.5, horizon: int = 1) -> pd.Series:
    """Posterior expected returns (per day) blending the equilibrium prior with signal views.

    Prior pi = delta * Sigma * w_mkt. Each asset gets an absolute view Q_i = prior_i + alpha_i/horizon with
    uncertainty omega_ii = (1/conf - 1) * tau * Sigma_ii (higher confidence -> smaller omega).
    """
    S = cov.to_numpy()
    pi = risk_aversion * S @ mkt_w.reindex(cov.index).fillna(0).to_numpy()
    Q = pi + view_alpha.reindex(cov.index).fillna(0).to_numpy() / horizon
    omega = np.diag((1 / view_conf - 1) * tau * np.diag(S))
    ts = tau * S
    post = pi + ts @ np.linalg.solve(ts + omega, Q - pi)
    return pd.Series(post, index=cov.index)


def mean_variance(mu: pd.Series, cov: pd.DataFrame, *, risk_aversion: float = 10.0, gross: float = 1.0,
                  net_max: float = 0.05, max_w: float = 0.1, w_prev: pd.Series | None = None,
                  turnover_penalty: float = 0.0, betas: pd.Series | None = None, beta_max: float = 0.1,
                  sectors: pd.Series | None = None, sector_net_max: float = 0.1) -> pd.Series:
    """`mu`/`cov` are daily. max mu'w - (lambda/2) w'Sigma w - tc*|w - w_prev|_1   s.t. |w|_1 <= gross, |sum w| <= net_max,
    |beta'w| <= beta_max, |w_i| <= max_w, |sector net| <= sector_net_max.  w = p - m with p, m >= 0."""
    names = list(cov.index)
    n = len(names)
    S = cov.to_numpy() * 252  # annualise: daily-scale objectives are ~1e-6 and SLSQP stops at its tolerance
    m0 = mu.reindex(names).fillna(0).to_numpy() * 252
    w0 = np.zeros(n) if w_prev is None else w_prev.reindex(names).fillna(0).to_numpy()
    b = None if betas is None else betas.reindex(names).fillna(1.0).to_numpy()

    def split(x):
        return x[:n] - x[n:]

    def obj(x):
        w = split(x)
        return -(m0 @ w) + 0.5 * risk_aversion * (w @ S @ w) + turnover_penalty * np.abs(w - w0).sum()

    cons = [{"type": "ineq", "fun": lambda x: gross - x.sum()},
            {"type": "ineq", "fun": lambda x: net_max - split(x).sum()},
            {"type": "ineq", "fun": lambda x: net_max + split(x).sum()}]
    if b is not None:
        cons += [{"type": "ineq", "fun": lambda x: beta_max - b @ split(x)},
                 {"type": "ineq", "fun": lambda x: beta_max + b @ split(x)}]
    if sectors is not None:
        sec = sectors.reindex(names).fillna("?")
        for s in sec.unique():
            mask = (sec == s).to_numpy().astype(float)
            cons += [{"type": "ineq", "fun": lambda x, mk=mask: sector_net_max - mk @ split(x)},
                     {"type": "ineq", "fun": lambda x, mk=mask: sector_net_max + mk @ split(x)}]
    res = minimize(obj, np.zeros(2 * n), method="SLSQP", bounds=[(0, max_w)] * (2 * n), constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-10})
    w = split(res.x)
    w[np.abs(w) < 1e-5] = 0.0
    return pd.Series(w, index=names)
