"""Structured LLM extraction with content-hash caching, self-consistency and entity verification."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import statistics
from collections import Counter, defaultdict

from stockpred.config.universe import Universe
from stockpred.data.entities import EntityLinker
from stockpred.data.models import Article, Extraction, TickerView
from stockpred.data.store import Store
from stockpred.nlp.providers import LLMProvider
from stockpred.nlp.schema import EVENT_TYPES, ArticleOut, parse_output

log = logging.getLogger(__name__)
PROMPT_VERSION = "v2"

_PROMPT = """\
You are a careful equity news analyst. Read the article and output ONE JSON object, nothing else.

Article:
{article}

Watchlist (symbol - company):
{watchlist}

JSON fields:
- "tickers": list of objects {{"ticker": <watchlist symbol>, "sentiment": <number -1..1>, "relevance": <number 0..1>}}.
  Include a symbol ONLY if the article is directly about that company's fundamentals or prospects.
  sentiment: -1 very negative for the share price, 0 neutral/unclear, +1 very positive.
  relevance: how central the company is to the article.
- "event_type": one of {event_types}
- "horizon": one of "intraday", "short" (1-5 days), "medium" (1-4 weeks), "long" (months+)
- "confidence": number 0..1, how clearly the article implies a price-relevant impact. Use low values for
  opinion pieces, rehashes, or vague items.
- "rationale": at most 20 words.

Judge only from the article text. If no watchlist company is directly affected, use "tickers": [].
"""

def build_prompt(article_text: str, universe: Universe) -> str:
    wl = "\n".join(f"{s.symbol} - {s.name}" for s in universe.securities)
    return _PROMPT.format(article=article_text, watchlist=wl, event_types=", ".join(EVENT_TYPES))


def cache_key(model: str, article_text: str, universe: Universe) -> str:
    h = hashlib.sha256()
    for part in (model, PROMPT_VERSION, article_text, ",".join(sorted(universe.symbols))):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def aggregate_samples(
    samples: list[ArticleOut], allowed: set[str], text_tickers: set[str] | None
) -> tuple[list[TickerView], str, str, float, str, float]:
    """Merge n samples -> (tickers, event_type, horizon, confidence, rationale, agreement)."""
    n = len(samples)
    sent: dict[str, list[float]] = defaultdict(list)
    rel: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        seen = set()
        for t in s.tickers:
            if t.ticker not in allowed or t.ticker in seen:
                continue
            if text_tickers is not None and t.ticker not in text_tickers:
                continue  # hallucinated / not evidenced in the text
            seen.add(t.ticker)
            sent[t.ticker].append(t.sentiment)
            rel[t.ticker].append(t.relevance)
    tickers, stds = [], []
    for tk, vals in sent.items():
        padded = vals + [0.0] * (n - len(vals))  # a sample that omitted the ticker voted "neutral"
        tickers.append(TickerView(tk, sum(padded) / n, sum(rel[tk]) / len(rel[tk])))
        if n > 1:
            stds.append(statistics.pstdev(padded))
    agreement = 1.0 if not stds else max(0.0, 1.0 - sum(stds) / len(stds))
    conf = sum(s.confidence for s in samples) / n
    ev = Counter(s.event_type for s in samples).most_common(1)[0][0]
    hz = Counter(s.horizon for s in samples).most_common(1)[0][0]
    rat = next((s.rationale for s in samples if s.rationale), "")
    return tickers, ev, hz, conf * agreement, rat, agreement


class Extractor:
    def __init__(
        self,
        provider: LLMProvider,
        store: Store,
        universe: Universe,
        *,
        temperature: float = 0.1,
        n_samples: int = 1,
        max_concurrent: int = 2,
        verify_entities: bool = True,
    ):
        self.provider = provider
        self.store = store
        self.universe = universe
        self.temperature = temperature
        self.n_samples = n_samples
        self.linker = EntityLinker(universe) if verify_entities else None
        self._sem = asyncio.Semaphore(max_concurrent)

    async def _sample(self, prompt: str, temp: float) -> ArticleOut | None:
        try:
            raw = await self.provider.generate(prompt, temperature=temp)
            return parse_output(raw)
        except Exception as exc:  # parse or transport; this sample is simply dropped
            log.warning("sample failed (%s): %s", self.provider.model, exc)
            return None

    async def extract(self, art: Article) -> Extraction:
        text = art.to_prompt_text()
        key = cache_key(self.provider.model, text, self.universe)
        hit = self.store.get_cached_extraction(key)
        if hit is not None:
            hit.article_id = art.id
            self.store.save_extraction(hit)
            return hit

        prompt = build_prompt(text, self.universe)
        temps = [self.temperature] + [max(self.temperature, 0.7)] * (self.n_samples - 1)
        async with self._sem:
            outs = await asyncio.gather(*[self._sample(prompt, t) for t in temps])
        good = [o for o in outs if o is not None]
        if not good:
            # failure => NOT cached or persisted, so the next cycle retries
            return Extraction(art.id, key, self.provider.model, "other", "short", 0.0,
                              "extraction failed", ok=False)

        text_tickers = self.linker.link(art.text, art.hint_ticker) if self.linker else None
        tickers, ev, hz, conf, rat, agree = aggregate_samples(
            good, set(self.universe.symbols), text_tickers
        )
        ex = Extraction(art.id, key, self.provider.model, ev, hz, conf, rat, tickers,
                        n_samples=len(good), agreement=agree, ok=True)
        self.store.save_extraction(ex)
        return ex

    async def extract_batch(self, arts: list[Article]) -> list[Extraction]:
        return list(await asyncio.gather(*[self.extract(a) for a in arts]))


def mask_entities(text: str, universe: Universe, token: str = "[COMPANY]") -> str:
    """Ablation helper: hide company identity so the LLM cannot recall post-publication outcomes.
    Compare model IC with/without masking to estimate look-ahead contamination."""
    out = text
    for s in universe.securities:
        for n in sorted({s.name, *s.aliases, s.symbol}, key=len, reverse=True):
            if len(n) >= 2:
                out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", token, out, flags=re.I)
    return out
