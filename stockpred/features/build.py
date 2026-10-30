"""Assemble the modelling panel: features + point-in-time-safe targets, indexed (date, symbol)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockpred.config.universe import Universe
from stockpred.data.macro import load_factors, yield_curve_features
from stockpred.data.store import Store
from stockpred.features.cross import cs_rank, factor_exposures, rolling_beta, sector_relative
from stockpred.features.regime import regime_prob_series
from stockpred.features.technical import adjust_ohlc, technical_features
from stockpred.features.volatility import volatility_features

log = logging.getLogger(__name__)
TARGET_PREFIX = ("fwd_", "tb_")
RANK_COLS = ["ret_5", "ret_21", "mom_12_1", "rv_20", "volume_z_20", "rsi_14", "dist_52w_high"]


def forward_returns(px: pd.DataFrame, horizon: int) -> pd.Series:
    """Tradeable label: buy at next open (t+1), sell at open (t+1+h), all split/dividend adjusted.

    Row t = information known at close t; the order fills at the t+1 open, so no
    same-bar fill is assumed.
    """
    a = adjust_ohlc(px)
    o = a["open"]
    return (o.shift(-(1 + horizon)) / o.shift(-1) - 1.0).rename(f"fwd_ret_{horizon}d")


def build_symbol_frame(px: pd.DataFrame, horizon: int, garch: bool = True) -> pd.DataFrame:
    px = px.sort_index()
    f = pd.concat([technical_features(px), volatility_features(px, garch=garch)], axis=1)
    f[f"fwd_ret_{horizon}d"] = forward_returns(px, horizon)
    f["fwd_ret_1d"] = forward_returns(px, 1)
    return f


def build_panel(
    store: Store,
    universe: Universe,
    horizon: int = 5,
    benchmark: str = "SPY",
    *,
    text_features: pd.DataFrame | None = None,
    garch: bool = True,
    regime: bool = True,
    use_factors: bool = False,
    start=None,
) -> pd.DataFrame:
    """Returns panel with feature columns plus targets:
    fwd_ret_{h}d (raw), fwd_exc_{h}d (minus benchmark), fwd_up_{h}d (excess > 0)."""
    symbols = [s for s in universe.symbols]
    px_all = store.prices_df(symbols + [benchmark], start=start)
    if px_all.empty:
        raise RuntimeError("no prices in store; run `stockpred ingest-prices`")
    bench_px = px_all[px_all.symbol == benchmark].set_index("date").sort_index()
    frames, rets = [], {}
    for sym, g in px_all[px_all.symbol.isin(symbols)].groupby("symbol"):
        g = g.set_index("date").sort_index()
        if len(g) < 300:
            log.warning("skip %s: only %d bars", sym, len(g))
            continue
        f = build_symbol_frame(g, horizon, garch=garch)
        f["symbol"] = sym
        frames.append(f.reset_index())
        rets[sym] = np.log(adjust_ohlc(g)["close"]).diff()
    if not frames:
        raise RuntimeError("no symbol has enough history")
    panel = pd.concat(frames).set_index(["date", "symbol"]).sort_index()

    # benchmark-relative target + market features
    bench_fwd = forward_returns(bench_px, horizon) if not bench_px.empty else None
    if bench_fwd is not None:
        b = bench_fwd.reindex(panel.index.get_level_values("date")).to_numpy()
        panel[f"fwd_exc_{horizon}d"] = panel[f"fwd_ret_{horizon}d"].to_numpy() - b
        panel[f"fwd_up_{horizon}d"] = (panel[f"fwd_exc_{horizon}d"] > 0).astype(float).where(
            panel[f"fwd_exc_{horizon}d"].notna())
        mkt_ret = np.log(adjust_ohlc(bench_px)["close"]).diff()
        panel["beta_mkt"] = np.nan
        for sym, r in rets.items():
            beta = rolling_beta(r, mkt_ret)
            ix = panel.index.get_level_values("symbol") == sym
            panel.loc[ix, "beta_mkt"] = beta.reindex(panel.index[ix].get_level_values("date")).to_numpy()
        mkt_feats = pd.DataFrame({"mkt_ret_5": mkt_ret.rolling(5).sum(),
                                  "mkt_ret_21": mkt_ret.rolling(21).sum(),
                                  "mkt_rv_20": mkt_ret.rolling(20).std() * np.sqrt(252)})
        if regime:
            mkt_feats["p_high_vol"] = regime_prob_series(mkt_ret)
        panel = _join_by_date(panel, mkt_feats)
    else:
        log.warning("benchmark %s missing: no excess-return target", benchmark)

    # macro + Fama-French residuals (optional: only when ingested)
    macro = yield_curve_features(store).shift(1)  # FRED values land with a lag: use t-1 only
    if not macro.empty and macro.notna().any().any():
        panel = _join_by_date(panel, macro)
    # Ken French factors are published weeks late: research-only, never use for live signals.
    ff = load_factors(store) if use_factors else pd.DataFrame()
    if not ff.empty and {"Mkt_RF", "SMB", "HML"} <= set(ff.columns):
        fcols = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in ff.columns]
        rf = ff["RF"] if "RF" in ff else None
        parts = []
        for sym, r in rets.items():
            ex = factor_exposures(np.expm1(r), ff[fcols], rf=rf)
            ex["symbol"] = sym
            parts.append(ex.reset_index(names="date").set_index(["date", "symbol"]))
        panel = panel.join(pd.concat(parts), how="left")

    # cross-sectional views
    present = [c for c in RANK_COLS if c in panel.columns]
    sec = pd.Series({s.symbol: s.sector for s in universe.securities})
    panel = panel.join(cs_rank(panel, present)).join(sector_relative(panel, ["ret_5", "ret_21"], sec))

    if text_features is not None:
        panel = panel.join(text_features, how="left")
    return panel.replace([np.inf, -np.inf], np.nan)


def _join_by_date(panel: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Join date-indexed features; values at date t use only data through t (ffill never looks ahead)."""
    d = daily.reindex(panel.index.get_level_values("date").unique().union(daily.index)).sort_index().ffill()
    d = d.reindex(panel.index.get_level_values("date"))
    d.index = panel.index
    return panel.join(d, how="left")


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if not c.startswith(TARGET_PREFIX)]
