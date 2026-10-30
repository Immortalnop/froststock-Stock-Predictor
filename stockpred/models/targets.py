"""Label construction: cross-sectional rank targets, triple-barrier labels, meta-labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.features.technical import adjust_ohlc


def cs_rank_target(y: pd.Series) -> pd.Series:
    """Per-date percentile rank of the target, centred (robust to fat tails / market-wide moves)."""
    return y.groupby(level="date").rank(pct=True) - 0.5


def winsorize_cs(y: pd.Series, q: float = 0.01) -> pd.Series:
    lo = y.groupby(level="date").transform(lambda s: s.quantile(q))
    hi = y.groupby(level="date").transform(lambda s: s.quantile(1 - q))
    return y.clip(lo, hi)


def triple_barrier(
    px: pd.DataFrame, horizon: int = 5, up_mult: float = 1.0, dn_mult: float = 1.0,
    vol_span: int = 20,
) -> pd.DataFrame:
    """Triple-barrier labels for one symbol (date-indexed raw OHLC + adj_close frame).

    Entry = adjusted open at t+1. Barriers at entry*(1 +/- mult * sigma_t * sqrt(horizon)),
    sigma_t = EWM daily vol through t. Path = days t+1 .. t+horizon (high/low). If both barriers
    are touched inside one bar the stop is assumed hit first (conservative).
    Returns columns tb_label (+1/-1/0 vertical), tb_ret (realised), tb_days (bars to exit).
    """
    a = adjust_ohlc(px)
    n = len(a)
    o, h, lo, c = (a[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sigma = np.log(a["close"]).diff().ewm(span=vol_span, min_periods=vol_span // 2).std().to_numpy()
    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    days = np.full(n, np.nan)
    for t in range(n - 1 - horizon):
        entry = o[t + 1]
        sg = sigma[t]
        if not np.isfinite(sg) or not np.isfinite(entry) or entry <= 0:
            continue
        w = sg * np.sqrt(horizon)
        up, dn = entry * (1 + up_mult * w), entry * (1 - dn_mult * w)
        lab, r, k_exit = 0, c[t + horizon] / entry - 1.0, horizon
        for k in range(1, horizon + 1):
            j = t + k
            hit_dn, hit_up = lo[j] <= dn, h[j] >= up
            if hit_dn:  # checked first: conservative on same-bar double touch
                lab, r, k_exit = -1, dn / entry - 1.0, k
                break
            if hit_up:
                lab, r, k_exit = 1, up / entry - 1.0, k
                break
        label[t], ret[t], days[t] = lab, r, k_exit
    return pd.DataFrame({"tb_label": label, "tb_ret": ret, "tb_days": days}, index=a.index)


def meta_labels(primary_side: pd.Series, tb_label: pd.Series) -> pd.Series:
    """1 if the primary model's side (+1/-1) matched the barrier outcome, else 0.
    Rows with side == 0 or vertical-barrier outcomes (0) count as failures to trade."""
    side = np.sign(primary_side)
    ok = (side * tb_label) > 0
    return ok.astype(float).where(tb_label.notna() & (side != 0))
