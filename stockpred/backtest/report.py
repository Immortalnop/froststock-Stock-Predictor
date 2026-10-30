"""Run a suite of candidate signals through the backtester and report with overfitting defences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.backtest.costs import ZERO_COST
from stockpred.backtest.engine import BacktestConfig, BacktestResult, market_arrays, run_backtest
from stockpred.backtest.metrics import (bootstrap_sharpe_ci, deflated_sharpe, factor_attribution, pbo_cscv,
                                        perf_stats, reality_check)
from stockpred.models.tracking import n_trials_total


@dataclass
class SuiteResult:
    table: pd.DataFrame
    results: dict[str, BacktestResult]
    overfit: dict = field(default_factory=dict)


def benchmark_returns(prices: pd.DataFrame, symbol: str = "SPY") -> pd.Series:
    op, _, _ = market_arrays(prices, [symbol])
    return (op[symbol].shift(-1) / op[symbol] - 1.0).rename(symbol)


def run_suite(signals: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig, *, benchmark: str = "SPY",
              factors: pd.DataFrame | None = None, n_trials: int | None = None, runs_dir: Path | None = None,
              n_boot: int = 500) -> SuiteResult:
    """signals: columns = candidate scores indexed by (date, symbol).

    n_trials defaults to the logged `n_trials_total` plus the number of candidates here, so the
    deflated Sharpe counts every configuration ever evaluated.
    """
    if n_trials is None:
        n_trials = (n_trials_total(runs_dir) if runs_dir else 0) + signals.shape[1]
    bench = benchmark_returns(prices, benchmark)
    fac = pd.DataFrame({"MKT": bench})
    if factors is not None and not factors.empty:
        fac = fac.join(factors.drop(columns=[c for c in factors.columns if c.lower() in ("rf", "mkt_rf", "mkt-rf")],
                                    errors="ignore"), how="left")
    results, rows = {}, {}
    gross_cfg = BacktestConfig(**{**cfg.__dict__, "cost": ZERO_COST})
    for name in signals.columns:
        s = signals[name].dropna()
        if s.empty:
            continue
        net = run_backtest(s, prices, cfg)
        if net.net_ret.empty:
            continue
        results[name] = net
        g = perf_stats(run_backtest(s, prices, gross_cfg).net_ret)
        n = perf_stats(net.net_ret)
        dsr = deflated_sharpe(net.net_ret, n_trials)
        lo, hi = bootstrap_sharpe_ci(net.net_ret, n_boot=n_boot)
        att = factor_attribution(net.net_ret, fac.dropna(axis=1, how="all"))
        rows[name] = {"gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"], "sharpe_lo": lo, "sharpe_hi": hi,
                      "ann_ret": n["ann_ret"], "ann_vol": n["ann_vol"], "max_dd": n["max_dd"],
                      "sortino": n["sortino"], "turnover_ann": net.avg_annual_turnover,
                      "cost_drag_ann": float((net.cost + net.borrow).mean() * 252),
                      "psr0": dsr["psr0"], "dsr": dsr["dsr"], "alpha_ann": att["alpha_ann"],
                      "alpha_t": att["alpha_t"], "beta_mkt": att.get("beta_MKT", np.nan), "n": n["n"]}
    table = pd.DataFrame(rows).T.sort_values("net_sharpe", ascending=False)
    mat = pd.DataFrame({k: v.net_ret for k, v in results.items()})
    overfit = {"n_trials": n_trials, **pbo_cscv(mat), **reality_check(mat, n_boot=n_boot)} if len(mat.columns) else {}
    return SuiteResult(table, results, overfit)


def capacity_curve(signal: pd.Series, prices: pd.DataFrame, cfg: BacktestConfig,
                   aums=(1e5, 1e6, 1e7, 1e8, 1e9)) -> pd.DataFrame:
    """Net Sharpe as a function of AUM: market impact grows with participation."""
    out = []
    for a in aums:
        c = BacktestConfig(**{**cfg.__dict__, "aum": float(a)})
        r = run_backtest(signal, prices, c)
        st = perf_stats(r.net_ret)
        out.append({"aum": a, "net_sharpe": st["sharpe"], "ann_ret": st["ann_ret"]})
    return pd.DataFrame(out).set_index("aum")


def decay_curve(signal: pd.Series, prices: pd.DataFrame, cfg: BacktestConfig,
                holds=(1, 2, 5, 10, 21)) -> pd.DataFrame:
    """Gross and net Sharpe vs holding period (rebalance interval): how fast does the signal decay?"""
    out = []
    for h in holds:
        c = BacktestConfig(**{**cfg.__dict__, "rebalance_days": int(h)})
        r = run_backtest(signal, prices, c)
        out.append({"hold_days": h, "net_sharpe": perf_stats(r.net_ret)["sharpe"],
                    "gross_sharpe": perf_stats(r.gross_ret)["sharpe"], "turnover_ann": r.avg_annual_turnover})
    return pd.DataFrame(out).set_index("hold_days")


def save_report(suite: SuiteResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"table": suite.table.to_dict("index"), "overfit": suite.overfit}, indent=2,
                               default=float))
    return path
