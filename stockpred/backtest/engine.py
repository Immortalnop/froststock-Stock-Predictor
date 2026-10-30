"""Event-ordered, cost-aware long/short backtester on daily bars.

Timing: a score dated d uses information up to the close of d. The order fills at the open of
the next bar (d+1) and the position earns open-to-open returns from then on. Weights drift
between rebalances. Names with no tradeable open are excluded and flattened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockpred.backtest.costs import CostModel


@dataclass
class BacktestConfig:
    rebalance_days: int = 5
    quantile: float = 0.2  # fraction of names on each side
    scheme: str = "quantile"  # "quantile" (equal weight) or "rank" (weight ~ demeaned rank)
    long_short: bool = True
    gross: float = 1.0  # sum |w|
    smooth: float = 1.0  # partial adjustment toward target each rebalance (1 = full)
    aum: float = 1_000_000.0  # for impact: trade dollars / ADV
    cost: CostModel = field(default_factory=CostModel)
    min_names: int = 8


@dataclass
class BacktestResult:
    gross_ret: pd.Series
    net_ret: pd.Series
    cost: pd.Series
    borrow: pd.Series
    turnover: pd.Series  # one-way, fraction of NAV traded per day
    weights: pd.DataFrame

    @property
    def avg_annual_turnover(self) -> float:
        return float(self.turnover.sum() / max(len(self.turnover), 1) * 252)


def target_weights(scores: pd.Series, cfg: BacktestConfig) -> pd.Series:
    """Cross-sectional target weights for one date (index = symbol, NaN scores excluded)."""
    s = scores.dropna()
    n = len(s)
    w = pd.Series(0.0, index=scores.index)
    if n < cfg.min_names:
        return w
    if cfg.scheme == "rank":
        r = s.rank() - (n + 1) / 2
        if not cfg.long_short:
            r = r.clip(lower=0)
        tot = r.abs().sum()
        if tot > 0:
            w.loc[r.index] = r / tot * cfg.gross
        return w
    k = max(1, int(round(cfg.quantile * n)))
    order = s.sort_values(kind="mergesort")
    if cfg.long_short:
        w.loc[order.index[-k:]] = cfg.gross / 2 / k
        w.loc[order.index[:k]] = -cfg.gross / 2 / k
    else:
        w.loc[order.index[-k:]] = cfg.gross / k
    return w


def market_arrays(prices: pd.DataFrame, symbols: list[str]):
    """Wide adjusted-open, ADV$ and daily-vol frames (ADV/vol lagged one bar: known at the fill)."""
    p = prices[prices.symbol.isin(symbols)].copy()
    p["open_adj"] = p["open"] * (p["adj_close"] / p["close"])
    p["dollar_vol"] = p["close"] * p["volume"]
    op = p.pivot(index="date", columns="symbol", values="open_adj").sort_index()
    cl = p.pivot(index="date", columns="symbol", values="adj_close").reindex(op.index)
    dv = p.pivot(index="date", columns="symbol", values="dollar_vol").reindex(op.index)
    adv = dv.rolling(20, min_periods=5).median().shift(1)
    vol = cl.pct_change().rolling(20, min_periods=5).std().shift(1)
    return op, adv, vol


def run_backtest(scores: pd.Series, prices: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    """scores: Series indexed by (date, symbol); higher = more bullish. prices: store.prices_df()."""
    cfg = cfg or BacktestConfig()
    sc = scores.dropna().unstack("symbol").sort_index()
    op, adv, vol = market_arrays(prices, list(sc.columns))
    sc = sc.reindex(columns=op.columns)
    dates = op.index
    oo = (op.shift(-1) / op - 1.0)  # return earned holding open(t) -> open(t+1), stored at t
    S = sc.reindex(dates)
    sig_dates = set(sc.index)

    n_sym = op.shape[1]
    w = np.zeros(n_sym)
    W, gross, cost_s, borrow_s, turn = [], [], [], [], []
    last_reb = -10**9
    oo_v, adv_v, vol_v = oo.to_numpy(), adv.to_numpy(), vol.to_numpy()
    bd = cfg.cost.borrow_daily()
    first = None
    for j, d in enumerate(dates):
        # signal from the previous bar's close, executed at this bar's open
        trade = np.zeros(n_sym)
        if j > 0 and dates[j - 1] in sig_dates:
            sig_i = j - 1
            if sig_i - last_reb >= cfg.rebalance_days:
                tgt = target_weights(S.iloc[sig_i], cfg).to_numpy()
                new = (1 - cfg.smooth) * w + cfg.smooth * tgt
                tradable = np.isfinite(oo_v[j])
                new = np.where(tradable, new, 0.0)  # cannot hold what has no next open
                trade, w = new - w, new
                last_reb = sig_i
                first = first if first is not None else j
        r = np.nan_to_num(oo_v[j], nan=0.0)
        if first is None or j < first:
            continue
        tc = 0.0
        if np.any(trade != 0):
            cf = cfg.cost.cost_frac(np.abs(trade) * cfg.aum, adv_v[j], vol_v[j])
            tc = float(np.sum(np.abs(trade) * cf))
        pr = float(np.dot(w, r))
        bw = bd * float(-w[w < 0].sum()) if np.any(w < 0) else 0.0
        gross.append(pr), cost_s.append(tc), borrow_s.append(bw), turn.append(float(np.abs(trade).sum() / 2))
        W.append(w.copy())
        # drift to next open
        nav = 1.0 + pr
        w = w * (1 + r) / nav if nav > 0 else w
    idx = dates[first:first + len(gross)] if first is not None else dates[:0]
    g = pd.Series(gross, idx, name="gross")
    c = pd.Series(cost_s, idx, name="cost")
    b = pd.Series(borrow_s, idx, name="borrow")
    net = (g - c - b).rename("net")
    # last bar has no next open: drop it
    keep = oo.reindex(idx).notna().any(axis=1).to_numpy()
    wdf = pd.DataFrame(W, index=idx, columns=op.columns)
    return BacktestResult(g[keep], net[keep], c[keep], b[keep], pd.Series(turn, idx, name="turnover")[keep], wdf[keep])
