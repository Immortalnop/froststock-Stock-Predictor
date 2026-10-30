"""Benchmarks every learned model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(panel: pd.DataFrame) -> pd.Series:
    return panel["mom_12_1"]


def reversal_5d(panel: pd.DataFrame) -> pd.Series:
    return -panel["ret_5"]


def sentiment_only(panel: pd.DataFrame, col: str = "sent_ew5d") -> pd.Series:
    return panel[col]


def legacy_score(panel: pd.DataFrame) -> pd.Series:
    """Approximation of the v0 hand-weighted composite from point-in-time features.

    v0:  0.50*sent + 0.20*buzz*sign(sent) + 0.15*recency*sent + 0.15*tanh(momentum)
    Here buzz = events / cross-sectional max events that day (the original relative-buzz flaw,
    kept on purpose so the benchmark is faithful), recency = 1 when there was news today else 0.5.
    """
    sent = panel["sent_ew1d"]
    n = panel["n_events_5d"]
    mx = n.groupby(level="date").transform("max").replace(0, np.nan)
    buzz = (n / mx).fillna(0.0)
    rec = np.where(panel["n_events_1d"] > 0, 1.0, 0.5)
    mom = np.tanh(25.0 * panel["ret_5"].fillna(0.0))
    score = 0.50 * sent + 0.20 * buzz * np.sign(sent) + 0.15 * rec * sent + 0.15 * mom
    return score.clip(-1, 1)


BASELINES = {
    "momentum_12_1": momentum_12_1,
    "reversal_5d": reversal_5d,
}
TEXT_BASELINES = {
    "sentiment_only": sentiment_only,
    "legacy_v0_score": legacy_score,
}


def available_baselines(panel: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for name, f in BASELINES.items():
        try:
            out[name] = f(panel)
        except KeyError:
            pass
    if "sent_ew5d" in panel and panel["n_events_5d"].fillna(0).sum() > 0:
        for name, f in TEXT_BASELINES.items():
            try:
                out[name] = f(panel)
            except KeyError:
                pass
    return out


def buy_and_hold_excess(panel: pd.DataFrame, target: str) -> float:
    """Equal-weight universe mean of the benchmark-relative return (IC is undefined for a constant)."""
    return float(panel[target].mean())
