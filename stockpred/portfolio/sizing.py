"""Position-sizing overlays: volatility targeting, fractional Kelly, drawdown throttle."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scale(port_returns: pd.Series, target_vol: float = 0.10, lookback: int = 60,
                     max_leverage: float = 3.0, periods: int = 252) -> pd.Series:
    """Leverage multiplier known at the close of t (uses returns up to and including t)."""
    realized = port_returns.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(periods)
    return (target_vol / realized).clip(upper=max_leverage).rename("vol_scale")


def kelly_fraction(mu: float, var: float, fraction: float = 0.25, cap: float = 1.0) -> float:
    """Fractional Kelly for a continuous return stream: f* = mu / sigma^2, scaled and capped.

    Full Kelly is ruinous under estimation error, so 1/4 Kelly is the default and negative edge sizes to 0.
    """
    if var <= 0 or not np.isfinite(mu):
        return 0.0
    return float(np.clip(fraction * mu / var, 0.0, cap))


def drawdown_throttle(equity: pd.Series, start: float = 0.05, full: float = 0.15) -> pd.Series:
    """Linear de-risking: 1 at drawdown <= start, 0 at drawdown >= full."""
    dd = 1 - equity / equity.cummax()
    return (1 - (dd - start) / (full - start)).clip(0.0, 1.0).rename("dd_scale")
