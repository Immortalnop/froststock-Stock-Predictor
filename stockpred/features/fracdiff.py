"""Fixed-width-window fractional differentiation (Lopez de Prado, AFML ch.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ffd_weights(d: float, thres: float = 1e-4, max_len: int = 2000) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_len:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # oldest weight first


def frac_diff_ffd(s: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """Uses only the trailing window, so value at t depends on data <= t."""
    w = ffd_weights(d, thres)
    width = len(w)
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= width:
        win = np.lib.stride_tricks.sliding_window_view(x, width)
        out[width - 1 :] = win @ w
    return pd.Series(out, index=s.index, name=f"{s.name}_fd{d:g}")


def min_d_stationary(s: pd.Series, grid=np.linspace(0.1, 1.0, 10), p_thresh: float = 0.05,
                     thres: float = 1e-3) -> float:
    """Smallest d in grid whose FFD series passes ADF (keeps maximum memory)."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    for d in grid:
        fd = frac_diff_ffd(s, float(d), thres).dropna()
        if len(fd) < 100:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pval = adfuller(fd.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
        if pval < p_thresh:
            return float(d)
    return 1.0
