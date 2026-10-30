"""Scheduled jobs and the shared runtime context used by the API and the scheduler."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.config.settings import Settings
from stockpred.config.universe import Universe
from stockpred.data.store import Store
from stockpred.mlops.drift import drift_report, retrain_decision
from stockpred.models.train import Bundle, load_bundle
from stockpred.serve.util import clean

log = logging.getLogger(__name__)
MIN_DRIFT_DATES = 10


@dataclass
class Runtime:
    cfg: Settings
    store: Store
    universe: Universe
    bundle: Bundle | None = None
    cache: dict = field(default_factory=dict)
    paper_enabled: bool = False
    paper_ignore_gate: bool = False
    auto_retrain: bool = False
    monitor: dict = field(default_factory=dict)

    def load_bundle(self) -> Bundle | None:
        try:
            self.bundle = load_bundle(self.cfg.artifacts_dir / "models")
        except Exception:
            self.bundle = None
        return self.bundle


def monitor_path(cfg: Settings) -> Path:
    return cfg.artifacts_dir / "monitor.json"


def run_monitor(rt: Runtime) -> dict:
    """Feature drift vs the training window, live ledger performance, retrain decision. Saved to monitor.json."""
    from stockpred.serve.ledger import live_performance, realized_ledger

    from stockpred.serve.pipeline import price_panel_cached

    cfg = rt.cfg
    out: dict = {"checked_at": pd.Timestamp.now("UTC").isoformat()}
    b = rt.bundle or rt.load_bundle()
    drift = pd.DataFrame(columns=["drift"])
    if b is not None:
        # features the model saw (<= trained_through) vs data that arrived after training: right after a
        # (re)train there is nothing to compare, so a fresh model never triggers its own retrain
        panel = price_panel_cached(rt.store, rt.universe, cfg, rt.cache)
        dates = panel.index.get_level_values("date")
        ref, cur = panel[dates <= b.trained_through], panel[dates > b.trained_through]
        out["n_dates_since_training"] = int(cur.index.get_level_values("date").nunique()) if len(cur) else 0
        if out["n_dates_since_training"] >= MIN_DRIFT_DATES:
            drift = drift_report(ref, cur, b.features)
            out["drift_share"] = float(drift["drift"].mean()) if len(drift) else None
            out["drifted_features"] = drift.index[drift["drift"].astype(bool)].tolist()[:15]
    ledger = realized_ledger(rt.store, cfg.horizon_days, cfg.benchmark)
    live = live_performance(ledger, cfg.horizon_days)
    out["live"] = live
    age = (pd.Timestamp.now() - b.trained_through).days if b is not None else np.inf
    out["model_age_days"] = None if b is None else int(age)
    retrain, why = retrain_decision(drift, age, live.get("ic_t_hac"), live.get("n_dates", 0))
    out["retrain"], out["retrain_reasons"] = retrain, why
    rt.monitor = out
    monitor_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    monitor_path(cfg).write_text(json.dumps(clean(out), indent=2))
    return out


def retrain(rt: Runtime) -> Path:
    """Rebuild the panel and refit the bundle with the same settings the CLI uses."""
    from stockpred.features.build import build_panel
    from stockpred.models.train import TrainConfig, fit_bundle, run_experiment, save_bundle
    from stockpred.serve.pipeline import text_feature_frame

    cfg = rt.cfg
    dates = pd.DatetimeIndex(sorted(rt.store.prices_df([cfg.benchmark]).date.unique()))
    src = "llm" if len(rt.store.extraction_rows()) else "lexicon"
    tf = text_feature_frame(rt.store, rt.universe, dates, src)
    panel = build_panel(rt.store, rt.universe, cfg.horizon_days, cfg.benchmark, text_features=tf)
    panel.to_parquet(cfg.artifacts_dir / "panel.parquet")
    tc = TrainConfig(horizon=cfg.horizon_days, embargo=cfg.embargo_days, seed=cfg.random_seed)
    res = run_experiment(panel, tc, cfg.runs_dir)
    path = save_bundle(fit_bundle(panel, tc, res), cfg.artifacts_dir / "models")
    rt.load_bundle()
    return path


def latest_targets(rt: Runtime):
    """(score, wide returns, betas, sectors, ic_t, as-of date) for portfolio construction."""
    from stockpred.serve.pipeline import price_panel_cached, signals_table

    pp = price_panel_cached(rt.store, rt.universe, rt.cfg, rt.cache)
    sig = signals_table(rt.store, rt.universe, pp, rt.bundle)
    px = rt.store.prices_df(rt.universe.symbols)
    rets = px.pivot(index="date", columns="symbol", values="adj_close").pct_change()
    last = pp.index.get_level_values("date").max()
    betas = pp.xs(last, level="date")["beta_mkt"] if "beta_mkt" in pp else None
    sectors = pd.Series({s: rt.universe.sector_of(s) for s in rt.universe.symbols})
    t = None if rt.bundle is None else rt.bundle.metrics.get("stack", {}).get("ic_t_hac")
    return sig["score"], rets, betas, sectors, t, last


def paper_cycle(rt: Runtime) -> dict:
    """Settle yesterday's orders, mark NAV, then queue the next rebalance."""
    from stockpred.portfolio.construct import ConstructConfig, build_target
    from stockpred.portfolio.paper import PaperAccount

    acct = PaperAccount(rt.store)
    filled = acct.settle()
    score, rets, betas, sectors, t, last = latest_targets(rt)
    acct.mark(last)
    cur = acct.positions()
    nav = acct.nav(last)
    px = rt.store.prices_df(end=last)
    cl = px[px.date == px.date.max()].set_index("symbol")["close"]
    w_prev = (cur * cl.reindex(cur.index)).div(nav) if len(cur) else None
    cons = build_target(score, rets, t, ConstructConfig(horizon=rt.cfg.horizon_days, ignore_gate=rt.paper_ignore_gate),
                        w_prev=w_prev, betas=betas, sectors=sectors)
    orders = acct.rebalance(cons.weights, last)
    return {"filled": filled, "queued": len(orders), "gate": cons.gate, "limits_ok": cons.report.ok,
            "violations": cons.report.violations, "nav": nav}


# ---------------------------------------------------------------- scheduler wiring
async def job_news(rt: Runtime) -> None:
    from stockpred.nlp.extractor import Extractor
    from stockpred.nlp.providers import make_provider
    from stockpred.serve.pipeline import extract_pending, ingest_news_once

    await ingest_news_once(rt.store, rt.universe, rt.cfg)
    prov = make_provider(rt.cfg)
    try:
        ok, _ = await prov.health()
        if ok:
            ex = Extractor(prov, rt.store, rt.universe, temperature=rt.cfg.llm_temperature,
                           n_samples=rt.cfg.llm_samples, max_concurrent=rt.cfg.llm_max_concurrent)
            await extract_pending(rt.store, ex, rt.cfg)
    finally:
        await prov.close()


async def job_signals(rt: Runtime) -> None:
    from stockpred.data.prices import ingest_prices_async
    from stockpred.serve.pipeline import log_signal_ledger, price_panel_cached, signals_table

    await ingest_prices_async(rt.store, rt.universe.symbols + [rt.cfg.benchmark], years=1)
    rt.bundle = rt.bundle or rt.load_bundle()

    def work():
        pp = price_panel_cached(rt.store, rt.universe, rt.cfg, rt.cache)
        sig = signals_table(rt.store, rt.universe, pp, rt.bundle)
        log_signal_ledger(rt.store, sig, rt.cfg.horizon_days, "stack" if rt.bundle else "untrained")
        if rt.paper_enabled:
            return paper_cycle(rt)

    await asyncio.to_thread(work)


async def job_monitor(rt: Runtime) -> None:
    out = await asyncio.to_thread(run_monitor, rt)
    if out.get("retrain") and rt.auto_retrain:
        log.warning("retraining: %s", out["retrain_reasons"])
        await asyncio.to_thread(retrain, rt)
