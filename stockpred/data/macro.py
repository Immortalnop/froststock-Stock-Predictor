"""Macro series (FRED CSV, no key) and Fama-French factors (Ken French library)."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from stockpred.data.store import Store

log = logging.getLogger(__name__)
FRED_SERIES = {"vix": "VIXCLS", "dgs10": "DGS10", "dgs2": "DGS2", "dgs3m": "DGS3MO"}
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "stockpred/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https URLs)
        return r.read()


def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    dcol, vcol = df.columns[0], df.columns[1]
    s = pd.to_numeric(df[vcol], errors="coerce")  # FRED uses "." for missing
    s.index = pd.to_datetime(df[dcol])
    return s.dropna().rename(vcol)


def fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    return parse_fred_csv(_get(url).decode("utf-8", "replace"))


def parse_french_csv(text: str) -> pd.DataFrame:
    """Daily Ken French CSV: header junk, then YYYYMMDD,val,val,... in percent."""
    rows, cols = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            break  # data block ended (annual section or footer)
        elif len(parts) > 1 and parts[0] == "":
            cols = [p for p in parts[1:] if p]
    if not rows:
        return pd.DataFrame()
    n = len(rows[0]) - 1
    names = cols if cols and len(cols) == n else [f"f{i}" for i in range(n)]
    df = pd.DataFrame([[float(x) for x in r[1:]] for r in rows], columns=names)
    df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    return df / 100.0


def fetch_french(url: str) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(_get(url)))
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    return parse_french_csv(z.read(name).decode("latin-1"))


def ingest_macro(store: Store) -> dict[str, int]:
    counts = {}
    for key, sid in FRED_SERIES.items():
        try:
            s = fetch_fred(sid)
            store.upsert_series(key, s)
            counts[key] = len(s)
        except Exception as exc:  # network/format; macro is optional
            log.warning("FRED %s failed: %s", sid, exc)
    try:
        ff = fetch_french(FF5_URL)
        try:
            ff = ff.join(fetch_french(MOM_URL).rename(columns=lambda c: "MOM"), how="left")
        except Exception as exc:
            log.warning("momentum factor failed: %s", exc)
        for c in ff.columns:
            store.upsert_series(f"ff_{c.replace('-', '_')}", ff[c])
        counts["french"] = len(ff)
    except Exception as exc:
        log.warning("Fama-French failed: %s", exc)
    return counts


def load_factors(store: Store) -> pd.DataFrame:
    """Daily FF factors from the store; columns Mkt_RF, SMB, HML, RMW, CMA, RF, MOM."""
    with store._lock:
        names = [r[0] for r in store.con.execute(
            "SELECT DISTINCT name FROM series WHERE name LIKE 'ff_%'").fetchall()]
    return pd.DataFrame({n[3:]: store.series(n) for n in names}).sort_index()


def yield_curve_features(store: Store) -> pd.DataFrame:
    d10, d2, d3m, vix = (store.series(k) for k in ("dgs10", "dgs2", "dgs3m", "vix"))
    df = pd.DataFrame({"dgs10": d10, "dgs2": d2, "dgs3m": d3m, "vix": vix}).sort_index().ffill()
    df["curve_10_2"] = df["dgs10"] - df["dgs2"]
    df["curve_10_3m"] = df["dgs10"] - df["dgs3m"]
    df["vix_chg_5d"] = df["vix"].pct_change(5)
    roll = df["vix"].rolling(252, min_periods=60)
    df["vix_z"] = (df["vix"] - roll.mean()) / roll.std().replace(0, np.nan)
    return df[["curve_10_2", "curve_10_3m", "vix", "vix_chg_5d", "vix_z"]]
