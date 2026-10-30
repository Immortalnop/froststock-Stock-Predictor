# core/predictor.py
"""
Aggregates LLM sentiment + price momentum into a composite prediction score.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.news_fetcher import Article
    from core.llm_analyzer import SentimentResult

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    symbol: str
    name: str
    sector: str

    current_price: float = 0.0
    price_change_pct: float = 0.0
    momentum_5d: float = 0.0

    raw_sentiment: float = 0.0
    sentiment_confidence: float = 0.0
    article_count: int = 0
    buzz_score: float = 0.0

    composite_score: float = 0.0
    prediction_label: str = "NEUTRAL"
    prediction_emoji: str = "⚪"
    top_rationale: str = ""

    last_updated: float = field(default_factory=time.time)
    contributing_ids: List[str] = field(default_factory=list)


_LABELS: List[Tuple[float, str, str]] = [
    ( 0.60, "STRONG BUY",  "🚀"),
    ( 0.25, "BUY",         "📈"),
    (-0.24, "NEUTRAL",     "⚪"),
    (-0.59, "SELL",        "📉"),
    (-1.01, "STRONG SELL", "🔴"),
]


def _score_to_label(score: float) -> Tuple[str, str]:
    for threshold, label, emoji in _LABELS:
        if score >= threshold:
            return label, emoji
    return "STRONG SELL", "🔴"


class Predictor:
    """
    Fuses four signals into a composite score per ticker:
      1. LLM sentiment          (configurable weight)
      2. Mention buzz           (normalised article count)
      3. Recency amplifier      (exponential decay, half-life 6 h)
      4. 5-day price momentum   (tanh-normalised OLS slope)
    """

    def __init__(
        self,
        watchlist: List[dict],
        weight_sentiment: float = 0.50,
        weight_buzz: float      = 0.20,
        weight_recency: float   = 0.15,
        weight_momentum: float  = 0.15,
    ):
        self.watchlist   = watchlist
        self.tickers     = [s["symbol"] for s in watchlist]
        self.ticker_meta = {s["symbol"]: s for s in watchlist}
        self.w_sent = weight_sentiment
        self.w_buzz = weight_buzz
        self.w_rec  = weight_recency
        self.w_mom  = weight_momentum

        self._signals: Dict[str, StockSignal] = {}
        self._price_cache: Dict[str, dict] = {}
        self._price_cache_ts: float = 0.0

    def _fetch_prices(self) -> Dict[str, dict]:
        now = time.time()
        if now - self._price_cache_ts < 300 and self._price_cache:
            return self._price_cache

        result: Dict[str, dict] = {
            sym: {"price": 0.0, "change_pct": 0.0, "momentum_5d": 0.0}
            for sym in self.tickers
        }

        try:
            import yfinance as yf
            batch = " ".join(self.tickers)
            all_tickers = yf.Tickers(batch)

            for sym in self.tickers:
                try:
                    hist = all_tickers.tickers[sym].history(period="6d", interval="1d")
                    if hist.empty:
                        continue
                    closes  = hist["Close"].to_numpy(dtype=float)
                    current = closes[-1]
                    prev    = closes[-2] if len(closes) > 1 else current
                    chg_pct = (current - prev) / prev * 100 if prev else 0.0

                    if len(closes) >= 5:
                        x = np.arange(5, dtype=float)
                        y = closes[-5:]
                        slope    = float(np.polyfit(x, y, 1)[0])
                        momentum = float(np.tanh(slope / max(current * 0.01, 1e-9)))
                    else:
                        momentum = 0.0

                    result[sym] = {
                        "price":       round(current, 2),
                        "change_pct":  round(chg_pct, 2),
                        "momentum_5d": round(momentum, 4),
                    }
                except Exception as exc:
                    logger.warning("Price fetch failed for %s: %s", sym, exc)

        except ImportError:
            logger.warning("yfinance not installed - price data unavailable.")
        except Exception as exc:
            logger.error("Batch price fetch error: %s", exc)

        self._price_cache    = result
        self._price_cache_ts = now
        return result

    def build_signals(
        self,
        sentiment_results: "List[SentimentResult]",
        articles: "List[Article]",
    ) -> Dict[str, StockSignal]:
        prices = self._fetch_prices()
        now    = time.time()

        ts_map: Dict[str, float] = {a.id: a.published_ts for a in articles}

        ticker_data: Dict[str, dict] = {
            sym: {"rows": [], "rationales": [], "ids": []}
            for sym in self.tickers
        }

        for res in sentiment_results:
            if not res.mentioned_tickers:
                continue
            age_h     = (now - ts_map.get(res.article_id, now)) / 3600
            recency_w = float(np.exp(-0.693 * age_h / 6))

            for sym, score in res.sentiment_scores.items():
                if sym not in ticker_data:
                    continue
                ticker_data[sym]["rows"].append(
                    (float(score), float(res.confidence), recency_w)
                )
                ticker_data[sym]["rationales"].append(res.rationale)
                ticker_data[sym]["ids"].append(res.article_id)

        max_mentions = max(
            (len(d["rows"]) for d in ticker_data.values()), default=1
        ) or 1

        signals: Dict[str, StockSignal] = {}

        for sym, meta in self.ticker_meta.items():
            td    = ticker_data[sym]
            rows  = td["rows"]
            price = prices.get(sym, {"price": 0.0, "change_pct": 0.0, "momentum_5d": 0.0})

            if rows:
                w_total = sum(c * r for _, c, r in rows)
                if w_total > 0:
                    raw_sent = sum(s * c * r for s, c, r in rows) / w_total
                    avg_conf = sum(c for _, c, _ in rows) / len(rows)
                    avg_rec  = sum(r for _, _, r in rows) / len(rows)
                else:
                    raw_sent = avg_conf = avg_rec = 0.0
            else:
                raw_sent = avg_conf = avg_rec = 0.0

            buzz      = len(rows) / max_mentions
            momentum  = float(np.clip(price["momentum_5d"], -1.0, 1.0))
            sent_sign = float(np.sign(raw_sent)) if raw_sent != 0 else 0.0

            composite = (
                self.w_sent * raw_sent
                + self.w_buzz * buzz * sent_sign
                + self.w_rec  * avg_rec * raw_sent
                + self.w_mom  * momentum
            )
            composite = float(np.clip(composite, -1.0, 1.0))

            label, emoji = _score_to_label(composite)
            top_rat = td["rationales"][0] if td["rationales"] else "No relevant news."

            signals[sym] = StockSignal(
                symbol=sym,
                name=meta["name"],
                sector=meta["sector"],
                current_price=price["price"],
                price_change_pct=price["change_pct"],
                momentum_5d=momentum,
                raw_sentiment=round(raw_sent, 3),
                sentiment_confidence=round(avg_conf, 3),
                article_count=len(rows),
                buzz_score=round(buzz, 3),
                composite_score=round(composite, 3),
                prediction_label=label,
                prediction_emoji=emoji,
                top_rationale=top_rat,
                contributing_ids=td["ids"][:5],
                last_updated=now,
            )

        self._signals = signals
        return signals

    def get_signals(self) -> Dict[str, StockSignal]:
        return self._signals

    def get_top_picks(self, n: int = 3) -> List[StockSignal]:
        return sorted(
            self._signals.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )[:n]

    def get_risk_alerts(self, threshold: float = -0.40) -> List[StockSignal]:
        return [
            s for s in self._signals.values()
            if s.composite_score <= threshold
        ]
