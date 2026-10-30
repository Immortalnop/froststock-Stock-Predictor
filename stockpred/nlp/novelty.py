"""Novelty = 1 - max cosine similarity to earlier articles in a trailing window.

TF-IDF by default (no model download). Only strictly *earlier* articles count, so the
score is point-in-time safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def novelty_scores(df: pd.DataFrame, window_hours: float = 72.0, text_col: str = "text",
                   time_col: str = "available_at") -> pd.Series:
    """df needs [id, text, available_at]. Returns Series indexed by id; first articles get 1.0."""
    if df.empty:
        return pd.Series(dtype=float)
    d = df.sort_values(time_col).reset_index(drop=True)
    X = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True).fit_transform(d[text_col])
    t = d[time_col].to_numpy("datetime64[s]").astype("int64")
    win = int(window_hours * 3600)
    out = np.ones(len(d))
    for i in range(1, len(d)):
        lo = np.searchsorted(t, t[i] - win, side="left")
        if lo >= i:
            continue
        sims = (X[lo:i] @ X[i].T).toarray().ravel()
        if sims.size:
            out[i] = 1.0 - float(sims.max())
    return pd.Series(out, index=d["id"].values)
