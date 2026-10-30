"""Market regime via 2-state Markov switching on returns, using *filtered* probabilities only."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def regime_prob_series(ret: pd.Series, refit_every: int = 126, min_obs: int = 750) -> pd.Series:
    """P(high-variance regime) at t from information through t.

    Parameters are re-estimated every `refit_every` days on past data; within a block the
    filter is run with those fixed parameters. Filtered (not smoothed) probabilities are
    causal, so there is no look-ahead.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    r = ret.dropna() * 100.0
    n = len(r)
    out = pd.Series(np.nan, index=r.index, name="p_high_vol")
    if n <= min_obs:
        return out.reindex(ret.index)
    vals = r.to_numpy()
    s = min_obs
    while s < n:
        end = min(s + refit_every, n)
        try:
            fit = MarkovRegression(vals[:s], k_regimes=2, switching_variance=True).fit(
                disp=False, maxiter=200
            )
            hi = int(np.argmax([fit.params[f"sigma2[{k}]"] for k in range(2)]))
            filt = MarkovRegression(vals[:end], k_regimes=2, switching_variance=True).filter(fit.params)
            p = np.asarray(filt.filtered_marginal_probabilities)[:, hi]
            out.iloc[s - 1 : end] = p[s - 1 : end]
        except Exception as exc:
            log.warning("regime fit failed at %d: %s", s, exc)
        s = end
    return out.reindex(ret.index)
