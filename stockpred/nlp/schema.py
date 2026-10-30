"""Validated LLM output schema. Everything is clipped/coerced; garbage raises ValidationError."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator

EVENT_TYPES = (
    "earnings", "guidance", "mna", "legal_regulatory", "product", "management",
    "analyst_rating", "macro", "capital_markets", "other",
)
HORIZONS = ("intraday", "short", "medium", "long")


def _num(v, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return max(lo, min(hi, x))


class TickerOut(BaseModel):
    ticker: str
    sentiment: float = 0.0
    relevance: float = 1.0

    @field_validator("ticker", mode="before")
    @classmethod
    def _t(cls, v):
        return str(v).strip().upper().replace(".", "-")

    @field_validator("sentiment", mode="before")
    @classmethod
    def _s(cls, v):
        return _num(v, -1.0, 1.0)

    @field_validator("relevance", mode="before")
    @classmethod
    def _r(cls, v):
        return _num(v, 0.0, 1.0, 1.0)


class ArticleOut(BaseModel):
    tickers: list[TickerOut] = Field(default_factory=list)
    event_type: str = "other"
    horizon: str = "short"
    confidence: float = 0.0
    rationale: str = ""

    @field_validator("event_type", mode="before")
    @classmethod
    def _e(cls, v):
        v = str(v).strip().lower()
        return v if v in EVENT_TYPES else "other"

    @field_validator("horizon", mode="before")
    @classmethod
    def _h(cls, v):
        v = str(v).strip().lower()
        return v if v in HORIZONS else "short"

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v):
        # missing/invalid confidence => 0, NOT a flattering 0.5 default
        return _num(v, 0.0, 1.0, 0.0)

    @field_validator("rationale", mode="before")
    @classmethod
    def _ra(cls, v):
        return str(v or "")[:300]

    @field_validator("tickers", mode="before")
    @classmethod
    def _tk(cls, v):
        """Accept [{ticker,sentiment,relevance}] or legacy {"AAPL": 0.5} / list of symbols."""
        if isinstance(v, dict):
            return [{"ticker": k, "sentiment": s} for k, s in v.items()]
        if isinstance(v, list):
            return [{"ticker": x} if isinstance(x, str) else x for x in v if x]
        return []


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of model output (handles fences / chatter)."""
    text = _FENCE.sub("", (raw or "").strip())
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            depth += c == "{"
            depth -= c == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return {}


def parse_output(raw: str) -> ArticleOut:
    data = extract_json(raw)
    if not data:
        raise ValueError("no JSON object in model output")
    return ArticleOut.model_validate(data)
