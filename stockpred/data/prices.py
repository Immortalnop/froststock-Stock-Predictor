"""Price ingestion (yfinance) into the store. Raw OHLC + adj_close are both kept.

Caveat: Yahoo restates adj_close after dividends/splits, so history is a snapshot.
Use `full=True` for a periodic complete refresh so adjusted series stay internally consistent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import pandas as pd

from stockpred.data.store import Store

log = logging.getLogger(__name__)
_COLS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


def normalize_yf(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Convert a yfinance multi-ticker (or single-ticker) frame to long format."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_COLS)
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        ticker_first = any(s in lvl0 for s in symbols)
        for s in symbols:
            try:
                sub = raw[s] if ticker_first else raw.xs(s, axis=1, level=1)
            except KeyError:
                continue
            frames.append(_one(sub, s))
    else:
        frames.append(_one(raw, symbols[0]))
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else None
    return out if out is not None and not out.empty else pd.DataFrame(columns=_COLS)


def _one(sub: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sub = sub.copy()
    sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
    if "adj_close" not in sub.columns:
        sub["adj_close"] = sub.get("close")
    need = ["open", "high", "low", "close", "adj_close", "volume"]
    if any(c not in sub.columns for c in need):
        return pd.DataFrame(columns=_COLS)
    sub = sub.dropna(subset=["close"])
    sub = sub[(sub["close"] > 0) & (sub["open"] > 0)]
    idx = pd.to_datetime(sub.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    sub.index = idx.normalize()
    sub = sub[~sub.index.duplicated(keep="last")]
    out = sub[need].reset_index(names="date")
    out.insert(0, "symbol", symbol)
    return out[_COLS]


def download(symbols: list[str], start: date, end: date | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbols, start=start.isoformat(), end=end.isoformat() if end else None,
        auto_adjust=False, group_by="ticker", progress=False,
        threads=False,  # yfinance's sqlite tz cache raises "database is locked" under threads
    )
    return normalize_yf(raw, symbols)


def download_with_retry(symbols: list[str], start: date, end: date | None = None) -> pd.DataFrame:
    """Batch download, then re-fetch any missing symbol one at a time (transient Yahoo failures)."""
    df = download(symbols, start, end)
    missing = sorted(set(symbols) - set(df["symbol"].unique()))
    parts = [df]
    for s in missing:
        parts.append(download([s], start, end))
    return pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else df


def ingest_prices(store: Store, symbols: list[str], years: float = 10, full: bool = False) -> int:
    """Incremental by default: resume 7 days before each symbol's last stored date."""
    today = date.today()
    default_start = today - timedelta(days=int(365.25 * years))
    total = 0
    groups: dict[date, list[str]] = {}
    for s in symbols:
        last = None if full else store.last_price_date(s)
        start = default_start if last is None else (last - pd.Timedelta(days=7)).date()
        groups.setdefault(start, []).append(s)
    for start, syms in groups.items():
        df = download_with_retry(syms, start)
        missing = set(syms) - set(df["symbol"].unique())
        if missing:
            log.warning("no price data for %s", sorted(missing))
        total += store.upsert_prices(df)
    return total


async def ingest_prices_async(*args, **kwargs) -> int:
    return await asyncio.to_thread(ingest_prices, *args, **kwargs)


def latest_quotes(store: Store, symbols: list[str]) -> dict[str, dict]:
    """Last close and 1-day change from stored daily bars."""
    df = store.prices_df(symbols)
    out: dict[str, dict] = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        if g.empty:
            continue
        px = float(g["close"].iloc[-1])
        prev = float(g["close"].iloc[-2]) if len(g) > 1 else px
        out[sym] = {"price": px, "change_pct": (px / prev - 1) * 100 if prev else 0.0,
                    "as_of": g["date"].iloc[-1]}
    return out
