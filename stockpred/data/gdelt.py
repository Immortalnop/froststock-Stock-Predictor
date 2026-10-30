"""GDELT DOC 2.0 backfill: free article metadata (titles/urls/dates) for ~the last 3 months.

GDELT returns no body text, only titles, so backfilled articles carry empty summaries.
`available_at` = seendate + assumed latency (no true ingestion time exists for history).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp

from stockpred.config.universe import Security
from stockpred.data.models import Article, make_article

log = logging.getLogger(__name__)
_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_seendate(s: str) -> datetime:
    # GDELT format: 20260101T123000Z
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ")


def parse_gdelt(payload: dict, sec: Security, latency: timedelta) -> list[Article]:
    out = []
    for it in payload.get("articles", []) or []:
        try:
            ts = parse_seendate(it["seendate"])
        except (KeyError, ValueError):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue
        out.append(
            make_article(title, "", it.get("url", ""), it.get("domain", "gdelt"), ts,
                         latency=latency, hint_ticker=sec.symbol)
        )
    return out


async def backfill_security(
    sess: aiohttp.ClientSession, sec: Security, days: int, latency: timedelta,
    max_records: int = 250,
) -> list[Article]:
    query = f'"{sec.name}" sourcelang:english'
    params = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": str(max_records), "timespan": f"{min(days, 90)}d", "sort": "datedesc",
    }
    for attempt in range(3):
        try:
            async with sess.get(_URL, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                if resp.status != 200:
                    return []
                return parse_gdelt(await resp.json(content_type=None), sec, latency)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            log.warning("gdelt %s: %s", sec.symbol, exc)
            await asyncio.sleep(2**attempt)
    return []


async def backfill(securities: list[Security], days: int, latency: timedelta) -> list[Article]:
    out: list[Article] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as sess:
        for sec in securities:  # sequential: GDELT rate-limits (~1 req / 5 s)
            out.extend(await backfill_security(sess, sec, days, latency))
            await asyncio.sleep(5)
    return out
