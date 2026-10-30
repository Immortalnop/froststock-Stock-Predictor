"""Point-in-time S&P 500 membership from Wikipedia's constituents + changes tables.

Using today's constituents for a 10-year backtest is survivorship bias (they are the winners).
`reconstruct_membership` rolls the current list backward through the change log.
"""

from __future__ import annotations

import io
import logging
import urllib.request

import pandas as pd

log = logging.getLogger(__name__)
_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def reconstruct_membership(current: set[str], changes: pd.DataFrame, asof) -> set[str]:
    """changes: columns date (datetime), added, removed (ticker or NaN). Roll back from today to `asof`."""
    asof = pd.Timestamp(asof)
    members = set(current)
    ch = changes.dropna(subset=["date"]).sort_values("date", ascending=False)
    for _, r in ch.iterrows():
        if r["date"] <= asof:
            break
        add, rem = r.get("added"), r.get("removed")
        if isinstance(add, str) and add:
            members.discard(add)
        if isinstance(rem, str) and rem:
            members.add(rem)
    return members


def fetch_tables() -> tuple[set[str], pd.DataFrame]:
    req = urllib.request.Request(_URL, headers={"User-Agent": "stockpred/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        html = r.read().decode("utf-8", "replace")
    tables = pd.read_html(io.StringIO(html))
    cur = tables[0]
    current = {str(s).replace(".", "-") for s in cur["Symbol"]}
    chg = tables[1].copy()
    chg.columns = ["_".join(str(x) for x in c if "Unnamed" not in str(x)).strip("_")
                   if isinstance(c, tuple) else str(c) for c in chg.columns]
    date_col = next(c for c in chg.columns if "Date" in c)
    add_col = next(c for c in chg.columns if c.startswith("Added") and "Ticker" in c)
    rem_col = next(c for c in chg.columns if c.startswith("Removed") and "Ticker" in c)
    out = pd.DataFrame({
        "date": pd.to_datetime(chg[date_col], errors="coerce"),
        "added": chg[add_col].astype("string").str.replace(".", "-", regex=False),
        "removed": chg[rem_col].astype("string").str.replace(".", "-", regex=False),
    })
    return current, out
