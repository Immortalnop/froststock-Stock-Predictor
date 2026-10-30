"""Prediction ledger: join logged predictions to realised excess returns -> live (true OOS) IC and hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.backtest.engine import market_arrays
from stockpred.data.store import Store
from stockpred.models.evaluate import hac_tstat, rank_ic

ET = "America/New_York"


def predictions_df(store: Store) -> pd.DataFrame:
    with store._lock:
        df = store.con.execute("SELECT * FROM predictions ORDER BY made_at").df()
    if df.empty:
        return df
    df["made_at"] = pd.to_datetime(df["made_at"])
    return df


def entry_dates(made_at: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Bar whose open the order fills at: the first open at or after `made_at` (naive UTC)."""
    et = made_at.dt.tz_localize("UTC").dt.tz_convert(ET)
    day = et.dt.tz_localize(None).dt.normalize()
    before_open = (et.dt.hour * 60 + et.dt.minute) < 9 * 60 + 30
    ix = trading_days.searchsorted(day, side="left")
    on_day = (ix < len(trading_days)) & (trading_days[np.minimum(ix, len(trading_days) - 1)] == day)
    ix = np.where(on_day & before_open, ix, trading_days.searchsorted(day, side="right"))
    ok = ix < len(trading_days)
    out = pd.Series(pd.NaT, index=made_at.index, dtype="datetime64[ns]")
    out[ok] = trading_days[ix[ok]]
    return out


def realized_ledger(store: Store, horizon: int, benchmark: str = "SPY") -> pd.DataFrame:
    """One row per (ET day, symbol): the last prediction of the day with its realised h-day excess return
    from the fill open to the open h bars later. Unrealised rows have NaN `realized`."""
    pr = predictions_df(store)
    if pr.empty:
        return pr
    pr = pr[pr.horizon_days == horizon].copy()
    if pr.empty:
        return pr
    px = store.prices_df()
    syms = sorted(set(pr.symbol) | {benchmark})
    op, _, _ = market_arrays(px, syms)
    days = op.index
    pr["entry"] = entry_dates(pr["made_at"], days)
    pr = pr.dropna(subset=["entry"]).sort_values("made_at")
    pr = pr.groupby(["entry", "symbol", "model"], as_index=False).last()
    pos = days.get_indexer(pr["entry"])
    ex = np.full(len(pr), np.nan)
    ok = pos + horizon < len(days)
    o_in = op.to_numpy()[pos[ok]]
    o_out = op.to_numpy()[pos[ok] + horizon]
    cols = op.columns.get_indexer(pr["symbol"])
    bcol = op.columns.get_loc(benchmark)
    r = o_out[np.arange(ok.sum()), cols[ok]] / o_in[np.arange(ok.sum()), cols[ok]] - 1
    b = o_out[:, bcol] / o_in[:, bcol] - 1
    ex[ok] = r - b
    pr["realized"] = ex
    return pr


def live_performance(ledger: pd.DataFrame, horizon: int) -> dict:
    """Rank IC (HAC t) and hit rate of realised predictions only."""
    if ledger.empty or ledger["realized"].notna().sum() == 0:
        return {"n_realized": 0, "n_dates": 0, "ic_mean": np.nan, "ic_t_hac": np.nan, "hit_rate": np.nan}
    d = ledger.dropna(subset=["realized"]).copy()
    d["date"] = pd.to_datetime(d["entry"])
    d = d.set_index(["date", "symbol"])
    ic = rank_ic(d["score"], d["realized"], min_names=5)
    hit = float((np.sign(d["score"] - d["score"].groupby(level="date").transform("mean"))
                 == np.sign(d["realized"] - d["realized"].groupby(level="date").transform("mean"))).mean())
    return {"n_realized": int(len(d)), "n_dates": int(len(ic)), "ic_mean": float(ic.mean()) if len(ic) else np.nan,
            "ic_t_hac": hac_tstat(ic, lags=max(1, horizon)) if len(ic) >= 10 else np.nan, "hit_rate": hit}
