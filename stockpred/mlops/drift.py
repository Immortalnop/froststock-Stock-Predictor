"""Feature drift (PSI, KS) and retrain triggers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < bins * 5 or len(cur) < 20:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(((q - p) * np.log(q / p)).sum())


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], psi_flag: float = 0.25,
                 ks_alpha: float = 0.001) -> pd.DataFrame:
    """Per-feature PSI and two-sample KS between the training window and the recent window."""
    rows = {}
    for f in features:
        if f not in ref or f not in cur:
            continue
        a, b = ref[f].to_numpy(float), cur[f].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 20:
            continue
        ks = stats.ks_2samp(a, b)
        p = psi(a, b)
        rows[f] = {"psi": p, "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
                   "drift": bool((p > psi_flag) or (ks.pvalue < ks_alpha and p > 0.1))}
    return pd.DataFrame(rows).T.sort_values("psi", ascending=False) if rows else pd.DataFrame(
        columns=["psi", "ks", "ks_p", "drift"])


@dataclass
class RetrainPolicy:
    max_model_age_days: int = 30
    max_drift_share: float = 0.2  # fraction of features flagged
    min_live_n: int = 60  # live observations before the live IC may trigger anything
    live_ic_t_floor: float = -2.0  # live IC significantly negative -> pull the model


def retrain_decision(drift: pd.DataFrame, model_age_days: float, live_ic_t: float | None, live_n: int,
                     policy: RetrainPolicy | None = None) -> tuple[bool, list[str]]:
    p = policy or RetrainPolicy()
    why = []
    if model_age_days > p.max_model_age_days:
        why.append(f"model age {model_age_days:.0f}d > {p.max_model_age_days}d")
    if len(drift) and drift["drift"].mean() > p.max_drift_share:
        why.append(f"{drift['drift'].mean():.0%} of features drifted (> {p.max_drift_share:.0%})")
    if live_n >= p.min_live_n and live_ic_t is not None and np.isfinite(live_ic_t) and live_ic_t < p.live_ic_t_floor:
        why.append(f"live IC t={live_ic_t:.1f} < {p.live_ic_t_floor}")
    return bool(why), why
