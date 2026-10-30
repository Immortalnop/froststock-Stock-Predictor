"""JSON helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python, timestamps -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o
