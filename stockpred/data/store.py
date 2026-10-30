"""DuckDB-backed point-in-time store.

Every news-derived row carries `available_at`; feature builders must filter on it so a
feature at time t only sees what was knowable at t.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from stockpred.data.models import Article, Extraction, TickerView, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id VARCHAR PRIMARY KEY,
    title VARCHAR, summary VARCHAR, url VARCHAR, source VARCHAR,
    published_at TIMESTAMP, ingested_at TIMESTAMP, available_at TIMESTAMP,
    hint_ticker VARCHAR, event_id VARCHAR
);
CREATE TABLE IF NOT EXISTS extractions (
    article_id VARCHAR, cache_key VARCHAR, model VARCHAR,
    event_type VARCHAR, horizon VARCHAR, confidence DOUBLE, rationale VARCHAR,
    n_samples INTEGER, agreement DOUBLE, extracted_at TIMESTAMP,
    PRIMARY KEY (article_id, cache_key)
);
CREATE TABLE IF NOT EXISTS extraction_tickers (
    article_id VARCHAR, cache_key VARCHAR, ticker VARCHAR,
    sentiment DOUBLE, relevance DOUBLE
);
CREATE TABLE IF NOT EXISTS prices (
    symbol VARCHAR, date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume DOUBLE,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS series (
    name VARCHAR, date DATE, value DOUBLE,
    PRIMARY KEY (name, date)
);
CREATE TABLE IF NOT EXISTS predictions (
    made_at TIMESTAMP, symbol VARCHAR, model VARCHAR, horizon_days INTEGER,
    score DOUBLE, p_up DOUBLE, lo DOUBLE, hi DOUBLE, label VARCHAR
);
CREATE TABLE IF NOT EXISTS paper_orders (
    order_id VARCHAR PRIMARY KEY, account VARCHAR, decided_on DATE, symbol VARCHAR,
    target_weight DOUBLE, qty DOUBLE, status VARCHAR, fill_date DATE, fill_price DOUBLE, cost DOUBLE
);
CREATE TABLE IF NOT EXISTS paper_positions (
    account VARCHAR, symbol VARCHAR, qty DOUBLE, PRIMARY KEY (account, symbol)
);
CREATE TABLE IF NOT EXISTS paper_nav (
    account VARCHAR, date DATE, nav DOUBLE, cash DOUBLE, gross DOUBLE, net DOUBLE,
    PRIMARY KEY (account, date)
);
CREATE TABLE IF NOT EXISTS paper_cash (
    account VARCHAR PRIMARY KEY, cash DOUBLE, initial DOUBLE
);
CREATE TABLE IF NOT EXISTS feed_health (
    feed VARCHAR, checked_at TIMESTAMP, ok BOOLEAN, n_items INTEGER, detail VARCHAR
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = duckdb.connect(str(path))
        with self._lock:
            self.con.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.con.close()

    # ------------------------------------------------------------ articles
    def upsert_articles(self, arts: Iterable[Article]) -> int:
        rows = [
            (a.id, a.title, a.summary, a.url, a.source, a.published_at, a.ingested_at,
             a.available_at, a.hint_ticker, a.event_id)
            for a in arts
        ]
        if not rows:
            return 0
        with self._lock:
            before = self.con.execute("SELECT count(*) FROM articles").fetchall()[0][0]
            self.con.executemany(
                "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING", rows
            )
            after = self.con.execute("SELECT count(*) FROM articles").fetchall()[0][0]
        return after - before

    def known_article_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        with self._lock:
            res = self.con.execute(
                "SELECT id FROM articles WHERE id IN (SELECT unnest(?))", [ids]
            ).fetchall()
        return {r[0] for r in res}

    def articles_df(
        self, since: datetime | None = None, available_by: datetime | None = None
    ) -> pd.DataFrame:
        q, args = "SELECT * FROM articles WHERE 1=1", []
        if since is not None:
            q += " AND published_at >= ?"
            args.append(since)
        if available_by is not None:
            q += " AND available_at <= ?"
            args.append(available_by)
        q += " ORDER BY published_at"
        with self._lock:
            return self.con.execute(q, args).df()

    def articles_pending(self, cache_prefix: str | None = None, limit: int = 200,
                         since: datetime | None = None) -> list[Article]:
        """Articles with no extraction yet (newest first)."""
        q = (
            "SELECT a.* FROM articles a WHERE NOT EXISTS "
            "(SELECT 1 FROM extractions e WHERE e.article_id = a.id)"
        )
        args: list = []
        if since is not None:
            q += " AND a.published_at >= ?"
            args.append(since)
        q += " ORDER BY a.published_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            df = self.con.execute(q, args).df()
        return [_row_to_article(r) for r in df.to_dict("records")]

    def set_event_ids(self, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        with self._lock:
            self.con.executemany(
                "UPDATE articles SET event_id = ? WHERE id = ?",
                [(ev, aid) for aid, ev in mapping.items()],
            )

    # ---------------------------------------------------------- extractions
    def get_cached_extraction(self, cache_key: str) -> Extraction | None:
        """Return a prior successful extraction with this cache key (any article)."""
        with self._lock:
            head = self.con.execute(
                "SELECT * FROM extractions WHERE cache_key = ? LIMIT 1", [cache_key]
            ).df()
            if head.empty:
                return None
            h = head.iloc[0]
            tk = self.con.execute(
                "SELECT ticker, sentiment, relevance FROM extraction_tickers "
                "WHERE cache_key = ? AND article_id = ?", [cache_key, h["article_id"]],
            ).fetchall()
        return Extraction(
            article_id=h["article_id"], cache_key=cache_key, model=h["model"],
            event_type=h["event_type"], horizon=h["horizon"], confidence=float(h["confidence"]),
            rationale=h["rationale"] or "", n_samples=int(h["n_samples"]),
            agreement=float(h["agreement"]),
            tickers=[TickerView(t, float(s), float(r)) for t, s, r in tk],
        )

    def save_extraction(self, ex: Extraction) -> None:
        """Persist a successful extraction. Failed ones (ok=False) are never stored."""
        if not ex.ok:
            return
        with self._lock:
            self.con.execute(
                "INSERT OR REPLACE INTO extractions VALUES (?,?,?,?,?,?,?,?,?,?)",
                [ex.article_id, ex.cache_key, ex.model, ex.event_type, ex.horizon,
                 ex.confidence, ex.rationale, ex.n_samples, ex.agreement, utcnow()],
            )
            self.con.execute(
                "DELETE FROM extraction_tickers WHERE article_id = ? AND cache_key = ?",
                [ex.article_id, ex.cache_key],
            )
            if ex.tickers:
                self.con.executemany(
                    "INSERT INTO extraction_tickers VALUES (?,?,?,?,?)",
                    [(ex.article_id, ex.cache_key, t.ticker, t.sentiment, t.relevance)
                     for t in ex.tickers],
                )

    def extraction_rows(self, model: str | None = None) -> pd.DataFrame:
        """One row per (article, ticker) with article timing + event_id + extraction fields."""
        q = """
        SELECT a.id AS article_id, a.event_id, a.published_at, a.available_at, a.source,
               e.model, e.event_type, e.horizon, e.confidence, e.agreement,
               t.ticker, t.sentiment, t.relevance, e.rationale
        FROM extraction_tickers t
        JOIN extractions e ON e.article_id = t.article_id AND e.cache_key = t.cache_key
        JOIN articles a ON a.id = t.article_id
        """
        args: list = []
        if model:
            q += " WHERE e.model = ?"
            args.append(model)
        with self._lock:
            return self.con.execute(q, args).df()

    # --------------------------------------------------------------- prices
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """df columns: symbol,date,open,high,low,close,adj_close,volume."""
        if df is None or df.empty:
            return 0
        d = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        with self._lock:
            self.con.register("_px", d)
            self.con.execute("INSERT OR REPLACE INTO prices SELECT * FROM _px")
            self.con.unregister("_px")
        return len(d)

    def prices_df(self, symbols: list[str] | None = None, start=None, end=None) -> pd.DataFrame:
        q, args = "SELECT * FROM prices WHERE 1=1", []
        if symbols:
            q += " AND symbol IN (SELECT unnest(?))"
            args.append(list(symbols))
        if start is not None:
            q += " AND date >= ?"
            args.append(pd.Timestamp(start).date())
        if end is not None:
            q += " AND date <= ?"
            args.append(pd.Timestamp(end).date())
        q += " ORDER BY symbol, date"
        with self._lock:
            df = self.con.execute(q, args).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def last_price_date(self, symbol: str):
        with self._lock:
            r = self.con.execute("SELECT max(date) FROM prices WHERE symbol = ?", [symbol]).fetchall()[0]
        return pd.Timestamp(r[0]) if r and r[0] is not None else None

    # --------------------------------------------------------------- series
    def upsert_series(self, name: str, s: pd.Series) -> None:
        s = s.dropna()
        if s.empty:
            return
        d = pd.DataFrame({"name": name, "date": pd.to_datetime(s.index).date, "value": s.values})
        with self._lock:
            self.con.register("_sr", d)
            self.con.execute("INSERT OR REPLACE INTO series SELECT * FROM _sr")
            self.con.unregister("_sr")

    def series(self, name: str) -> pd.Series:
        with self._lock:
            df = self.con.execute(
                "SELECT date, value FROM series WHERE name = ? ORDER BY date", [name]
            ).df()
        if df.empty:
            return pd.Series(dtype=float, name=name)
        return pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=name)

    # ----------------------------------------------------------- ledger etc
    def log_predictions(self, rows: list[tuple]) -> None:
        if rows:
            with self._lock:
                self.con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)", rows)

    def log_feed_health(self, feed: str, ok: bool, n_items: int, detail: str = "") -> None:
        with self._lock:
            self.con.execute(
                "INSERT INTO feed_health VALUES (?,?,?,?,?)", [feed, utcnow(), ok, n_items, detail]
            )

    def feed_health_df(self) -> pd.DataFrame:
        with self._lock:
            return self.con.execute(
                "SELECT feed, count(*) AS checks, avg(ok::INT) AS ok_rate, sum(n_items) AS items "
                "FROM feed_health GROUP BY feed ORDER BY feed"
            ).df()


def _row_to_article(r: dict) -> Article:
    def ts(x):
        return pd.Timestamp(x).to_pydatetime().replace(tzinfo=None)

    return Article(
        id=r["id"], title=r["title"], summary=r["summary"] or "", url=r["url"] or "",
        source=r["source"] or "", published_at=ts(r["published_at"]),
        ingested_at=ts(r["ingested_at"]), available_at=ts(r["available_at"]),
        hint_ticker=r.get("hint_ticker") or None, event_id=r.get("event_id") or None,
    )
