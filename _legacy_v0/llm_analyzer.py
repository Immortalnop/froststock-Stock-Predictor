# core/llm_analyzer.py
"""
LLM-powered sentiment extractor.
Primary backend: Ollama (fully local, no API key).
Fallback: any OpenAI-compatible endpoint.
Results are cached in memory to avoid redundant LLM calls.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from core.news_fetcher import Article

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────
_PROMPT = """\
You are a quantitative financial analyst AI.
Analyse the news article below and return ONLY a valid JSON object — no markdown fences, no extra text.

Article:
{article_text}

Watchlist tickers: {tickers}

Required JSON schema (fill every field):
{{
  "mentioned_tickers": ["AAPL"],
  "sentiment_scores":  {{"AAPL": 0.7}},
  "impact_horizon":    "short",
  "confidence":        0.8,
  "rationale":         "one-sentence reason"
}}

Rules:
- mentioned_tickers : only tickers from the watchlist DIRECTLY affected by this article.
- sentiment_scores  : float -1.0 (very bearish) ... +1.0 (very bullish) per ticker.
- impact_horizon    : "intraday" | "short" (1-5 d) | "medium" (1-4 wk) | "long" (months+).
- confidence        : 0.0-1.0 reflecting how clearly the article implies a market impact.
- rationale         : <= 20 words.
- If no ticker is relevant, return empty lists/dicts for those fields.
"""


# ─────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    article_id: str
    mentioned_tickers: List[str]
    sentiment_scores: Dict[str, float]
    impact_horizon: str
    confidence: float
    rationale: str
    raw_json: dict = field(default_factory=dict, repr=False)


_EMPTY = dict(
    mentioned_tickers=[],
    sentiment_scores={},
    impact_horizon="short",
    confidence=0.0,
    rationale="",
)


# ─────────────────────────────────────────────────────────────
# ANALYSER
# ─────────────────────────────────────────────────────────────
class LLMAnalyzer:
    """
    Async LLM client with in-memory TTL cache.
    At most `max_concurrent` simultaneous LLM calls.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 60,
        cache_ttl: int = 300,
        max_concurrent: int = 3,
        api_key: str = "",
    ):
        self.provider    = provider
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout     = timeout
        self.cache_ttl   = cache_ttl
        self.api_key     = api_key

        self._cache: Dict[str, tuple] = {}
        self._sem   = asyncio.Semaphore(max_concurrent)
        self._sess: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._sess is None or self._sess.closed:
            self._sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._sess

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[SentimentResult]:
        entry = self._cache.get(key)
        if entry:
            result, ts = entry
            if time.time() - ts < self.cache_ttl:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, result: SentimentResult):
        self._cache[key] = (result, time.time())

    async def _call_ollama(self, prompt: str) -> str:
        sess = await self._get_session()
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": self.temperature},
        }
        async with sess.post(f"{self.base_url}/api/generate", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Ollama {resp.status}: {body[:200]}")
            data = await resp.json()
        return data.get("response", "")

    async def _call_openai_compat(self, prompt: str) -> str:
        sess = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        async with sess.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            data = await resp.json()
        return data["choices"][0]["message"]["content"]

    async def _llm(self, prompt: str) -> str:
        if self.provider == "ollama":
            return await self._call_ollama(prompt)
        if self.provider in ("openai", "anthropic"):
            return await self._call_openai_compat(prompt)
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        text = text.strip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {}

    def _build_result(self, article_id: str, data: dict) -> SentimentResult:
        return SentimentResult(
            article_id=article_id,
            mentioned_tickers=data.get("mentioned_tickers", []),
            sentiment_scores={
                k: float(v)
                for k, v in data.get("sentiment_scores", {}).items()
            },
            impact_horizon=data.get("impact_horizon", "short"),
            confidence=float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "")),
            raw_json=data,
        )

    async def analyze_article(
        self,
        article_id: str,
        article_text: str,
        tickers: List[str],
    ) -> SentimentResult:
        key = self._cache_key(article_text[:500])
        cached = self._get_cached(key)
        if cached:
            logger.debug("Cache hit: %s", article_id)
            return cached

        prompt = _PROMPT.format(
            article_text=article_text[:1500],
            tickers=", ".join(tickers),
        )

        async with self._sem:
            try:
                raw    = await self._llm(prompt)
                data   = self._parse_json(raw)
                result = self._build_result(article_id, data)
            except Exception as exc:
                logger.error("LLM error for %s: %s", article_id, exc)
                result = SentimentResult(
                    article_id=article_id,
                    rationale=f"LLM error: {exc}",
                    **_EMPTY,
                )

        self._set_cache(key, result)
        return result

    async def analyze_batch(
        self,
        articles: "List[Article]",
        tickers: List[str],
    ) -> "List[SentimentResult]":
        tasks = [
            self.analyze_article(a.id, a.to_prompt_text(), tickers)
            for a in articles
        ]
        return await asyncio.gather(*tasks)

    async def check_ollama_health(self) -> bool:
        try:
            sess = await self._get_session()
            async with sess.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return False
                data  = await resp.json()
                names = [m["name"].split(":")[0] for m in data.get("models", [])]
                model_base = self.model.split(":")[0]
                if model_base in names:
                    return True
                logger.warning(
                    "Model '%s' not found. Available: %s", self.model, names
                )
        except Exception as exc:
            logger.error("Ollama health check failed: %s", exc)
        return False

    async def close(self):
        if self._sess and not self._sess.closed:
            await self._sess.close()
