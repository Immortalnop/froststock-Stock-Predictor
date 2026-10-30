"""Core data records. All datetimes are timezone-naive UTC."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

_TAG = re.compile(r"<[^>]*>")
_WS = re.compile(r"\s+")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def clean_text(raw: str) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    if not raw:
        return ""
    txt = _TAG.sub(" ", raw)
    txt = html.unescape(txt)
    return _WS.sub(" ", txt).strip()


def article_id(title: str, url: str) -> str:
    key = f"{title.strip().lower()}|{url.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


@dataclass
class Article:
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    ingested_at: datetime
    available_at: datetime
    hint_ticker: str | None = None
    event_id: str | None = None

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or utcnow()
        return (now - self.published_at).total_seconds() / 3600.0

    def to_prompt_text(self, max_summary: int = 600) -> str:
        return f"HEADLINE: {self.title}\nSUMMARY: {self.summary[:max_summary].strip()}"


def make_article(
    title: str,
    summary: str,
    url: str,
    source: str,
    published_at: datetime,
    *,
    ingested_at: datetime | None = None,
    latency: timedelta = timedelta(0),
    hint_ticker: str | None = None,
) -> Article:
    """Build an Article. `available_at` = when a live system could first have seen it.

    Live ingestion: available_at = max(published_at, ingested_at).
    Backfill (ingested_at is "now", far after publication): pass `ingested_at=None`
    and a `latency`, which gives available_at = published_at + latency.
    """
    title = clean_text(title)
    summary = clean_text(summary)
    ing = ingested_at or utcnow()
    avail = published_at + latency if ingested_at is None else max(published_at, ing)
    return Article(
        id=article_id(title, url),
        title=title,
        summary=summary,
        url=url,
        source=source,
        published_at=published_at,
        ingested_at=ing,
        available_at=avail,
        hint_ticker=hint_ticker,
    )


@dataclass
class TickerView:
    ticker: str
    sentiment: float  # -1..1
    relevance: float  # 0..1


@dataclass
class Extraction:
    """Structured LLM output for one article (validated and clipped)."""

    article_id: str
    cache_key: str
    model: str
    event_type: str
    horizon: str
    confidence: float
    rationale: str
    tickers: list[TickerView] = field(default_factory=list)
    n_samples: int = 1
    agreement: float = 1.0
    ok: bool = True
