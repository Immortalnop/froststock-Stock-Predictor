"""Async RSS/Atom ingestion with correct UTC handling and per-feed health stats."""

from __future__ import annotations

import asyncio
import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import aiohttp
import feedparser

from stockpred.config.universe import Universe
from stockpred.data.models import Article, make_article, utcnow
from stockpred.data.store import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Feed:
    url: str
    hint_ticker: str | None = None


GENERAL_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.marketwatch.com/rss/topstories",
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom",
    "https://techcrunch.com/feed/",
]


def yahoo_ticker_feed(symbol: str) -> Feed:
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={symbol}&region=US&lang=en-US"
    )
    return Feed(url, hint_ticker=symbol)


def default_feeds(universe: Universe) -> list[Feed]:
    return [Feed(u) for u in GENERAL_FEEDS] + [yahoo_ticker_feed(s.yahoo) for s in universe.securities]


def entry_time(entry) -> datetime | None:
    """feedparser gives UTC struct_time: use timegm (NOT mktime, which assumes local tz)."""
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if not st:
        return None
    return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc).replace(tzinfo=None)


def parse_feed(
    xml_text: str,
    feed: Feed,
    *,
    ingested_at: datetime | None = None,
    latency: timedelta = timedelta(0),
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> tuple[list[Article], int]:
    """Parse feed XML. Returns (articles, n_skipped_without_date).

    Entries without a publication date are dropped, never defaulted to "now"
    (that would make stale items look fresh and leak into recency features).
    """
    parsed = feedparser.parse(xml_text)
    source = parsed.feed.get("title") or urlparse(feed.url).netloc
    now = now or utcnow()
    out, skipped = [], 0
    for e in parsed.entries:
        ts = entry_time(e)
        if ts is None:
            skipped += 1
            continue
        if ts > now + timedelta(minutes=10):  # future-dated => clock/feed error
            skipped += 1
            continue
        if max_age_hours is not None and (now - ts).total_seconds() / 3600 > max_age_hours:
            continue
        title = (e.get("title") or "").strip()
        if not title:
            continue
        out.append(
            make_article(
                title, e.get("summary") or e.get("description") or "", e.get("link") or "",
                source, ts, ingested_at=ingested_at, latency=latency, hint_ticker=feed.hint_ticker,
            )
        )
    return out, skipped


class RssIngestor:
    def __init__(
        self,
        feeds: list[Feed],
        user_agent: str,
        store: Store | None = None,
        max_age_hours: float = 24.0,
        retries: int = 2,
    ):
        self.feeds = feeds
        self.user_agent = user_agent
        self.store = store
        self.max_age_hours = max_age_hours
        self.retries = retries
        self._sess: aiohttp.ClientSession | None = None

    async def _session(self) -> aiohttp.ClientSession:
        if self._sess is None or self._sess.closed:
            self._sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": self.user_agent, "Accept": "application/rss+xml, */*"},
            )
        return self._sess

    async def _get(self, url: str) -> str:
        sess = await self._session()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with sess.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    if resp.status not in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"HTTP {resp.status}")
                    last = RuntimeError(f"HTTP {resp.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last = exc
            await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError(str(last))

    async def _one(self, feed: Feed) -> list[Article]:
        try:
            xml = await self._get(feed.url)
            arts, skipped = parse_feed(xml, feed, ingested_at=utcnow(), max_age_hours=self.max_age_hours)
            if self.store:
                self.store.log_feed_health(feed.url, True, len(arts), f"skipped_nodate={skipped}")
            return arts
        except Exception as exc:
            log.warning("feed failed %s: %s", feed.url, exc)
            if self.store:
                self.store.log_feed_health(feed.url, False, 0, str(exc)[:200])
            return []

    async def fetch_new(self) -> list[Article]:
        """Fetch all feeds; return only articles not already in the store, deduped by id.

        Nothing is marked "seen" here: the store row is the single source of truth, and
        callers persist via Store.upsert_articles, so truncating the batch later cannot
        silently lose articles.
        """
        batches = await asyncio.gather(*[self._one(f) for f in self.feeds])
        by_id: dict[str, Article] = {}
        for b in batches:
            for a in b:
                by_id.setdefault(a.id, a)
        arts = list(by_id.values())
        if self.store and arts:
            known = self.store.known_article_ids([a.id for a in arts])
            arts = [a for a in arts if a.id not in known]
        arts.sort(key=lambda a: a.published_at, reverse=True)
        return arts

    async def close(self) -> None:
        if self._sess and not self._sess.closed:
            await self._sess.close()
