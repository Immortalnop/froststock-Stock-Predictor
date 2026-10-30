"""Price/volume features. Convention: row t uses data through the close of day t only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_ohlc(px: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by adj_close/close so splits/dividends do not create fake gaps."""
    f = px["adj_close"] / px["close"]
    out = pd.DataFrame(
        {"open": px["open"] * f, "high": px["high"] * f, "low": px["low"] * f,
         "close": px["adj_close"], "volume": px["volume"]},
        index=px.index,
    )
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def technical_features(px: pd.DataFrame) -> pd.DataFrame:
    """px: date-indexed frame with open/high/low/close/adj_close/volume (unadjusted + adj_close)."""
    a = adjust_ohlc(px)
    c = a["close"]
    f = pd.DataFrame(index=a.index)
    for h in (1, 5, 10, 21, 63, 126):
        f[f"ret_{h}"] = c.pct_change(h)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0  # skip most recent month
    f["rsi_14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["atr_pct"] = atr(a) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    f["boll_width"] = 4 * sd20 / ma20
    dv = (a["volume"] * c).replace(0, np.nan)
    ldv = np.log(dv)
    f["dollar_vol_log"] = ldv
    f["volume_z_20"] = (ldv - ldv.rolling(20).mean()) / ldv.rolling(20).std().replace(0, np.nan)
    f["gap"] = a["open"] / c.shift(1) - 1.0
    f["intraday_ret"] = c / a["open"] - 1.0
    f["dist_52w_high"] = c / c.rolling(252, min_periods=126).max() - 1.0
    f["dist_ma50"] = c / c.rolling(50).mean() - 1.0
    return f.replace([np.inf, -np.inf], np.nan)
