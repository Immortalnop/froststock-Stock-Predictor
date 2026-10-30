# core/news_fetcher.py
"""
Async RSS news fetcher with deduplication and age filtering.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import feedparser

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """A single news article."""
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_ts: float

    @property
    def age_hours(self) -> float:
        return (time.time() - self.published_ts) / 3600

    def to_prompt_text(self) -> str:
        summary = self.summary[:400].strip()
        return f"HEADLINE: {self.title}\nSUMMARY: {summary}"


class NewsFetcher:
    """
    Fetches and deduplicates articles from multiple RSS feeds concurrently.
    """

    def __init__(self, feeds: List[str], max_age_hours: float = 12):
        self.feeds = feeds
        self.max_age_hours = max_age_hours
        self._seen_ids: set = set()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "StockNewsBot/1.0"},
            )
        return self._session

    async def _fetch_feed(self, feed_url: str) -> List[Article]:
        articles: List[Article] = []
        try:
            sess = await self._get_session()
            async with sess.get(feed_url) as resp:
                if resp.status != 200:
                    logger.warning("Feed %s -> HTTP %s", feed_url, resp.status)
                    return []
                raw_xml = await resp.text(errors="replace")

            parsed = feedparser.parse(raw_xml)
            source = parsed.feed.get("title", feed_url.split("/")[2])

            for entry in parsed.entries:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                url     = entry.get("link", "")

                ts_struct = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                )
                published_ts = time.mktime(ts_struct) if ts_struct else time.time()

                art_id = hashlib.sha256(
                    f"{title}{url}".encode()
                ).hexdigest()[:16]

                if art_id in self._seen_ids:
                    continue

                art = Article(
                    id=art_id,
                    title=title,
                    summary=summary,
                    url=url,
                    source=source,
                    published_ts=published_ts,
                )

                if art.age_hours <= self.max_age_hours:
                    articles.append(art)
                    self._seen_ids.add(art_id)

        except asyncio.TimeoutError:
            logger.warning("Timeout on feed: %s", feed_url)
        except Exception as exc:
            logger.error("Error fetching %s: %s", feed_url, exc)

        return articles

    async def fetch_all(self) -> List[Article]:
        """Fetch all feeds concurrently; return newest-first."""
        tasks = [self._fetch_feed(url) for url in self.feeds]
        batches = await asyncio.gather(*tasks)
        all_articles: List[Article] = [a for batch in batches for a in batch]
        all_articles.sort(key=lambda a: a.published_ts, reverse=True)
        logger.info(
            "Fetched %d unique articles from %d feeds",
            len(all_articles), len(self.feeds),
        )
        return all_articles

    def reset_seen(self):
        """Clear dedup cache (call periodically to reprocess older articles)."""
        self._seen_ids.clear()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
