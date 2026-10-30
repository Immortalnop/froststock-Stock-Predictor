"""Calibrate LLM confidence against realized outcomes, and benchmark sentiment scorers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score


class ConfidenceCalibrator:
    """Maps raw LLM confidence -> P(sign(sentiment) == sign(realized excess return)).

    Fit on rows where the LLM took a directional view; only on data strictly before the
    period you evaluate on (caller's responsibility).
    """

    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        self.fitted = False

    def fit(self, confidence, sentiment, realized_ret) -> "ConfidenceCalibrator":
        c, s, r = (np.asarray(x, float) for x in (confidence, sentiment, realized_ret))
        m = (np.abs(s) > 1e-9) & (np.abs(r) > 1e-12) & np.isfinite(c) & np.isfinite(r)
        if m.sum() < 30:
            raise ValueError("need >= 30 directional observations to calibrate")
        hit = (np.sign(s[m]) == np.sign(r[m])).astype(float)
        self.iso.fit(c[m], hit)
        self.fitted = True
        return self

    def predict(self, confidence) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("not fitted")
        return self.iso.predict(np.asarray(confidence, float))


def to_label(score: float, thresh: float = 0.15) -> str:
    return "positive" if score > thresh else "negative" if score < -thresh else "neutral"


def evaluate_classifier(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=["negative", "neutral", "positive"],
                         zero_division=0)),
        "n": len(y_true),
    }


def load_phrasebank(path: str | Path) -> pd.DataFrame:
    """Financial PhraseBank file lines: `sentence@label` (labels positive/neutral/negative)."""
    rows = []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if "@" in line:
            s, lab = line.rsplit("@", 1)
            lab = lab.strip().lower()
            if lab in ("positive", "neutral", "negative"):
                rows.append((s.strip(), lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def benchmark_scorers(df: pd.DataFrame, scorers: dict[str, callable]) -> pd.DataFrame:
    """df has [text,label]; scorers map name -> f(text)->score in [-1,1]."""
    out = []
    for name, f in scorers.items():
        pred = [to_label(f(t)) for t in df["text"]]
        out.append({"scorer": name, **evaluate_classifier(df["label"].tolist(), pred)})
    return pd.DataFrame(out)
