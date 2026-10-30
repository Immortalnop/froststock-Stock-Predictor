"""Performance statistics and multiple-testing defences (PSR, deflated Sharpe, PBO, Reality Check)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


def perf_stats(r: pd.Series, periods: int = 252) -> dict[str, float]:
    """Annualised performance of a daily return series."""
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann_ret", "ann_vol", "sharpe", "sortino", "calmar", "max_dd", "skew",
                                    "kurt", "n")}
    mu, sd = r.mean() * periods, r.std() * np.sqrt(periods)
    dn = np.sqrt((np.minimum(r, 0) ** 2).mean() * periods)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"ann_ret": float(mu), "ann_vol": float(sd), "sharpe": float(mu / sd) if sd > 0 else np.nan,
            "sortino": float(mu / dn) if dn > 0 else np.nan,
            "calmar": float(mu / abs(dd)) if dd < 0 else np.nan, "max_dd": dd,
            "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False)), "n": int(len(r))}


def psr(sr: float, n: int, skew: float, kurt: float, sr_bench: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado). `sr` is the per-period Sharpe."""
    if n < 3 or not np.isfinite(sr):
        return np.nan
    den = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if den <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_bench) * np.sqrt(n - 1) / np.sqrt(den)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max of n independent Sharpe ratios] under the null, per-period units."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(var_sr) * ((1 - EULER) * z(1 - 1 / n_trials) + EULER * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r: pd.Series, n_trials: int, var_sr_trials: float | None = None) -> dict[str, float]:
    """DSR: probability the true Sharpe is > 0 after deflating by the best-of-N selection bias.

    `n_trials` = every configuration ever evaluated (`n_trials_total` in the run logs). If the
    cross-trial Sharpe variance is unknown it defaults to the sampling variance of one Sharpe
    estimate, 1/(n-1).
    """
    r = r.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return {"psr0": np.nan, "dsr": np.nan, "sr_star": np.nan, "n_trials": n_trials}
    sr = r.mean() / r.std()
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    v = var_sr_trials if var_sr_trials is not None else 1.0 / (n - 1)
    sr_star = expected_max_sharpe(n_trials, v)
    return {"psr0": psr(sr, n, sk, ku, 0.0), "dsr": psr(sr, n, sk, ku, sr_star),
            "sr_star": sr_star * np.sqrt(252), "n_trials": n_trials}


def stationary_bootstrap_idx(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices."""
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe_ci(r: pd.Series, n_boot: int = 1000, mean_block: int = 10, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float]:
    x = r.dropna().to_numpy()
    if len(x) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    sr = np.empty(n_boot)
    for b in range(n_boot):
        s = x[stationary_bootstrap_idx(len(x), mean_block, rng)]
        sr[b] = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0
    a = (1 - level) / 2
    return float(np.quantile(sr, a)), float(np.quantile(sr, 1 - a))


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 10, max_combos: int = 252, seed: int = 0) -> dict[str, float]:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

    `returns`: rows = time, columns = every candidate strategy tried. For each split of the time
    blocks into IS/OOS halves, pick the IS-best strategy and record its OOS relative rank.
    PBO = P(IS winner ranks at or below the OOS median).
    """
    R = returns.dropna(how="any")
    n_blocks -= n_blocks % 2
    if R.shape[1] < 2 or len(R) < n_blocks * 5:
        return {"pbo": np.nan, "n_combos": 0, "n_strategies": R.shape[1]}
    blocks = np.array_split(np.arange(len(R)), n_blocks)
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    X = R.to_numpy()

    def sharpe(rows):
        s = X[rows]
        sd = s.std(axis=0)
        return np.where(sd > 0, s.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)

    logits = []
    for c in combos:
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in c])
        best = int(np.argmax(sharpe(is_rows)))
        oos = sharpe(oos_rows)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1)
        w = (rank + 1) / (len(oos) + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": len(combos), "n_strategies": R.shape[1],
            "median_logit": float(np.median(logits))}


def reality_check(returns: pd.DataFrame, n_boot: int = 500, mean_block: int = 10, seed: int = 0) -> dict:
    """White Reality Check: is the best of K strategies better than zero after data snooping?

    `returns`: columns = K strategies' (excess-over-benchmark) return series. p-value from a
    stationary bootstrap of the centred statistic max_k sqrt(n) * mean_k.
    """
    R = returns.dropna(how="any")
    if R.shape[1] < 1 or len(R) < 50:
        return {"rc_pvalue": np.nan, "best": None}
    X = R.to_numpy()
    n = len(X)
    means = X.mean(axis=0)
    stat = float(np.sqrt(n) * means.max())
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_idx(n, mean_block, rng)
        cnt += np.sqrt(n) * (X[idx].mean(axis=0) - means).max() >= stat
    return {"rc_pvalue": float((cnt + 1) / (n_boot + 1)), "best": str(R.columns[int(np.argmax(means))])}


def factor_attribution(r: pd.Series, factors: pd.DataFrame, lags: int = 5, periods: int = 252) -> dict[str, float]:
    """Regress strategy returns on factor returns; annualised alpha with Newey-West t-stat."""
    import statsmodels.api as sm

    df = pd.concat([r.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < 60:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "r2": np.nan}
    res = sm.OLS(df["y"], sm.add_constant(df[list(factors.columns)])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha_ann": float(res.params["const"] * periods), "alpha_t": float(res.tvalues["const"]),
           "r2": float(res.rsquared)}
    for c in factors.columns:
        out[f"beta_{c}"] = float(res.params[c])
    return out
