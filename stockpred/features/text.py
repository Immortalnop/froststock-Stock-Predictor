"""News -> per-(date, symbol) features with exponentially decayed memory.

Point-in-time rule: an article only affects the feature row of the first trading day whose
16:00 New York close is at or after its `available_at`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.data.entities import EntityLinker
from stockpred.nlp.lexicon import score_text

NY = "America/New_York"
EVENT_FLAGS = ("earnings", "guidance", "mna", "legal_regulatory", "analyst_rating")


def assign_feature_date(available_at: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Map naive-UTC timestamps to the first trading date whose close (16:00 ET) has them in view."""
    ts = pd.to_datetime(available_at).dt.tz_localize("UTC").dt.tz_convert(NY)
    day = ts.dt.normalize().dt.tz_localize(None)
    after_close = (ts.dt.hour > 16) | ((ts.dt.hour == 16) & (ts.dt.minute > 0)) | (
        (ts.dt.hour == 16) & (ts.dt.minute == 0) & (ts.dt.second > 0)
    )
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    td = pd.DatetimeIndex(trading_dates).normalize()
    pos = td.searchsorted(day.to_numpy(), side="left")
    vals = np.where(pos < len(td), td.to_numpy()[np.minimum(pos, len(td) - 1)], np.datetime64("NaT"))
    return pd.Series(pd.to_datetime(vals), index=available_at.index)


def lexicon_rows(articles: pd.DataFrame, linker: EntityLinker) -> pd.DataFrame:
    """Build extraction-style rows from raw articles using the lexicon scorer (cheap, backfill-able)."""
    rows = []
    for r in articles.itertuples():
        text = f"{r.title}. {r.summary or ''}"
        syms = linker.link(text, getattr(r, "hint_ticker", None))
        if not syms:
            continue
        s = score_text(text)
        for sym in syms:
            rows.append({
                "article_id": r.id, "event_id": r.event_id or r.id,
                "available_at": r.available_at, "ticker": sym, "sentiment": s,
                "relevance": 1.0, "confidence": 1.0, "event_type": "other", "novelty": 1.0,
            })
    return pd.DataFrame(rows)


def build_text_features(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    symbols: list[str],
    *,
    novelty: pd.Series | None = None,
    halflives: tuple[float, ...] = (1.0, 5.0),
    prefix: str = "",
    clamp_to_last: bool = False,
) -> pd.DataFrame:
    """rows: [article_id,event_id,available_at,ticker,sentiment,relevance,confidence,event_type].

    Returns frame indexed (date, symbol) over trading_dates x symbols.
    """
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(trading_dates), symbols], names=["date", "symbol"])
    cols_zero = ["n_events", "sent_sum", "w_sum", "abs_sent", "sent_disp"] + [f"ev_{e}" for e in EVENT_FLAGS]
    if rows is None or rows.empty:
        base = pd.DataFrame(0.0, index=idx, columns=cols_zero)
    else:
        r = rows[rows["ticker"].isin(symbols)].copy()
        r["date"] = assign_feature_date(r["available_at"], trading_dates).to_numpy()
        if clamp_to_last:  # live row: everything after the last close belongs to the newest row
            r["date"] = r["date"].fillna(pd.Timestamp(trading_dates[-1]))
        r = r.dropna(subset=["date"])
        nov = 1.0
        if novelty is not None:
            nov = r["article_id"].map(novelty).fillna(1.0)
        elif "novelty" in r:
            nov = r["novelty"].fillna(1.0)
        r["w"] = r["confidence"] * r["relevance"] * (0.5 + 0.5 * nov)
        r["ws"] = r["w"] * r["sentiment"]
        # collapse syndicated copies: one observation per (date, symbol, event)
        g = r.groupby(["date", "ticker", "event_id"])
        ev = g.agg(w=("w", "max"), ws=("ws", "sum"), wt=("w", "sum"), s=("sentiment", "mean"),
                   etype=("event_type", lambda x: x.mode().iat[0]))
        ev["sent_ev"] = np.where(ev["wt"] > 0, ev["ws"] / ev["wt"].replace(0, np.nan), 0.0)
        ev["ws_ev"] = ev["w"] * ev["sent_ev"]
        ev = ev.reset_index().rename(columns={"ticker": "symbol"})
        d = ev.groupby(["date", "symbol"])
        base = pd.DataFrame({
            "n_events": d.size().astype(float),
            "sent_sum": d["ws_ev"].sum(),
            "w_sum": d["w"].sum(),
            "abs_sent": d["sent_ev"].apply(lambda x: float(np.abs(x).mean())),
            "sent_disp": d["sent_ev"].std(ddof=0),
        })
        for e in EVENT_FLAGS:
            base[f"ev_{e}"] = ev[ev["etype"] == e].groupby(["date", "symbol"]).size().astype(float)
        base = base.reindex(idx).fillna(0.0)

    out = pd.DataFrame(index=idx)
    for h in halflives:
        lam = 0.5 ** (1.0 / h)
        S = base["sent_sum"].unstack("symbol").ewm(alpha=1 - lam, adjust=False).mean() / (1 - lam)
        W = base["w_sum"].unstack("symbol").ewm(alpha=1 - lam, adjust=False).mean() / (1 - lam)
        out[f"sent_ew{h:g}d"] = (S / W.where(W > 1e-9)).fillna(0.0).stack().reindex(idx).to_numpy()
        out[f"sent_mass{h:g}d"] = S.stack().reindex(idx).to_numpy()
    ne = base["n_events"].unstack("symbol")
    n5 = ne.rolling(5, min_periods=1).sum()
    mu, sd = n5.rolling(60, min_periods=20).mean(), n5.rolling(60, min_periods=20).std()
    out["n_events_1d"] = ne.stack().reindex(idx).to_numpy()
    out["n_events_5d"] = n5.stack().reindex(idx).to_numpy()
    out["buzz_z"] = ((n5 - mu) / sd.replace(0, np.nan)).stack().reindex(idx).to_numpy()
    since = ne.copy()
    for c in since.columns:
        s, last = [], 9999.0
        for v in ne[c].to_numpy():
            last = 0.0 if v > 0 else min(last + 1, 9999.0)
            s.append(last)
        since[c] = s
    out["days_since_news"] = since.stack().reindex(idx).to_numpy()
    out["abs_sent"] = base["abs_sent"].to_numpy()
    out["sent_disp"] = base["sent_disp"].to_numpy()
    for e in EVENT_FLAGS:
        out[f"ev_{e}_3d"] = base[f"ev_{e}"].unstack("symbol").rolling(3, min_periods=1).sum() \
            .stack().reindex(idx).to_numpy()
    out["has_news_5d"] = (out["n_events_5d"] > 0).astype(float)
    if prefix:
        out.columns = [f"{prefix}{c}" for c in out.columns]
    return out
