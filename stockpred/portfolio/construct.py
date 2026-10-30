"""Signal -> target weights: alpha mapping, evidence gate, Black-Litterman blend, constrained MV, risk limits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockpred.portfolio.optimize import (black_litterman, evidence_scale, ledoit_wolf_cov, mean_variance,
                                          signal_to_alpha)
from stockpred.portfolio.risk import LimitReport, RiskLimits, check_limits, hist_var_cvar, scale_to_limits


@dataclass
class ConstructConfig:
    horizon: int = 5
    ic_assumed: float = 0.03  # used only after the evidence gate; typical equity cross-sectional IC
    cov_lookback: int = 252
    risk_aversion: float = 10.0
    turnover_penalty: float = 0.0005
    use_bl: bool = True
    view_conf: float = 0.5
    ignore_gate: bool = False  # measure the null: trade the signal even without walk-forward evidence
    limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class Construction:
    weights: pd.Series
    alpha: pd.Series
    gate: float
    report: LimitReport


def build_target(score: pd.Series, returns: pd.DataFrame, ic_t: float | None, cfg: ConstructConfig,
                 w_prev: pd.Series | None = None, betas: pd.Series | None = None,
                 sectors: pd.Series | None = None) -> Construction:
    """score: latest cross-sectional model score by symbol. returns: wide daily returns (dates x symbols)."""
    names = [s for s in score.dropna().index if s in returns.columns]
    R = returns[names].tail(cfg.cov_lookback).dropna(how="any", axis=1).dropna(how="any")
    names = list(R.columns)
    zero = pd.Series(0.0, index=score.index)
    gate = 1.0 if cfg.ignore_gate else evidence_scale(ic_t)
    if len(names) < 8 or gate == 0.0:
        return Construction(zero, zero, gate, check_limits(zero, cfg.limits))
    cov = ledoit_wolf_cov(R)
    vol = pd.Series(np.sqrt(np.diag(cov)), index=names)
    alpha = signal_to_alpha(score[names], vol, cfg.ic_assumed * gate, cfg.horizon)
    if cfg.use_bl:
        mkt = pd.Series(1.0 / len(names), index=names)  # equal-weight equilibrium proxy
        mu = black_litterman(cov, mkt, alpha, view_conf=cfg.view_conf, horizon=cfg.horizon)
        mu = mu - mkt.dot(mu)  # the optimiser is near-neutral; drop the common market premium
    else:
        mu = alpha / cfg.horizon
    lim = cfg.limits
    w = mean_variance(mu, cov, risk_aversion=cfg.risk_aversion, gross=lim.max_gross, net_max=lim.max_net,
                      max_w=lim.max_name, w_prev=w_prev, turnover_penalty=cfg.turnover_penalty, betas=betas,
                      beta_max=lim.max_beta, sectors=sectors, sector_net_max=lim.max_sector_net)
    pnl = (R * w).sum(axis=1)
    var95 = hist_var_cvar(pnl, 0.95)["var"]
    w = scale_to_limits(w, lim, var95).reindex(score.index).fillna(0.0)
    return Construction(w, alpha.reindex(score.index), gate, check_limits(w, lim, betas, sectors, var95))
