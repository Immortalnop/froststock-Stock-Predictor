"""Optional FinBERT baseline (needs `pip install transformers torch`). Imported lazily."""

from __future__ import annotations


class FinBertScorer:
    MODEL = "ProsusAI/finbert"

    def __init__(self, device: int = -1):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("FinBERT needs: pip install transformers torch") from exc
        self._pipe = pipeline("text-classification", model=self.MODEL, top_k=None, device=device)

    def score(self, texts: list[str]) -> list[float]:
        """Return P(positive) - P(negative) in [-1, 1] for each text."""
        out = []
        for res in self._pipe(texts, truncation=True, max_length=256):
            d = {r["label"].lower(): r["score"] for r in res}
            out.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        return out
