"""Minimal experiment tracking: one JSON file per run (params, metrics, data fingerprint).

`n_trials_total` counts every configuration ever evaluated, needed later for deflated-Sharpe
style multiple-testing corrections.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(str(list(df.columns)).encode())
    if len(df):
        h.update(str(df.index[0]).encode())
        h.update(str(df.index[-1]).encode())
        h.update(pd.util.hash_pandas_object(df.tail(200), index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def n_trials_total(runs_dir: Path) -> int:
    """Every configuration ever evaluated, for multiple-testing corrections."""
    return sum(json.loads(p.read_text()).get("n_trials", 1) for p in Path(runs_dir).glob("*.json"))


def log_run(runs_dir: Path, name: str, params: dict, metrics: dict, data_fp: str,
            n_trials: int = 1) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    prior = sum(json.loads(p.read_text()).get("n_trials", 1) for p in runs_dir.glob("*.json"))
    rec = {"name": name, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "params": params,
           "metrics": metrics, "data_fingerprint": data_fp, "n_trials": n_trials,
           "n_trials_total": prior + n_trials}
    path = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.json"
    path.write_text(json.dumps(rec, indent=2, default=float))
    return path
