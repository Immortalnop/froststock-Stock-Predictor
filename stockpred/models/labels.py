"""Signal labels derived from the *predicted distribution*, not hand-set score thresholds."""

from __future__ import annotations

import pandas as pd

LABEL_EMOJI = {"STRONG BUY": "🚀", "BUY": "📈", "NEUTRAL": "⚪", "SELL": "📉", "STRONG SELL": "🔴"}


def cross_sectional_labels(score: pd.Series, p_up: pd.Series | None = None) -> pd.Series:
    """Label by the percentile of today's predicted score within the universe.

    top 10% STRONG BUY, next 20% BUY, bottom 10% STRONG SELL, next 20% SELL. If a calibrated
    P(up) is supplied, a BUY needs P(up) >= 0.5 and a SELL needs P(up) <= 0.5, else NEUTRAL.
    """
    pct = score.rank(pct=True)
    lab = pd.Series("NEUTRAL", index=score.index)
    lab[pct >= 0.7] = "BUY"
    lab[pct >= 0.9] = "STRONG BUY"
    lab[pct <= 0.3] = "SELL"
    lab[pct <= 0.1] = "STRONG SELL"
    if p_up is not None:
        p = p_up.reindex(score.index)
        lab[lab.isin(["BUY", "STRONG BUY"]) & (p < 0.5)] = "NEUTRAL"
        lab[lab.isin(["SELL", "STRONG SELL"]) & (p > 0.5)] = "NEUTRAL"
    return lab
