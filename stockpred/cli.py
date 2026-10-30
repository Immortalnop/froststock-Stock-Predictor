"""Command line interface: `python -m stockpred <command>`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta

import pandas as pd

from stockpred.config import default_universe, get_settings
from stockpred.data.store import Store

log = logging.getLogger("stockpred")


def _setup(cfg):
    cfg.ensure_dirs()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(cfg.data_dir / "stockpred.log", encoding="utf-8")],
    )
    return Store(cfg.db_path)


def _panel_path(cfg):
    return cfg.artifacts_dir / "panel.parquet"


# ------------------------------------------------------------------ commands
def cmd_ingest_prices(a, cfg, store, uni):
    from stockpred.data.prices import ingest_prices

    syms = (a.symbols.split(",") if a.symbols else uni.symbols) + [cfg.benchmark]
    n = ingest_prices(store, sorted(set(syms)), years=a.years, full=a.full)
    print(f"upserted {n} price rows")


def cmd_ingest_macro(a, cfg, store, uni):
    from stockpred.data.macro import ingest_macro

    print(ingest_macro(store))


def cmd_ingest_news(a, cfg, store, uni):
    from stockpred.serve.pipeline import ingest_news_once

    new = asyncio.run(ingest_news_once(store, uni, cfg))
    print(f"{len(new)} new articles")
    print(store.feed_health_df().to_string(index=False))


def cmd_backfill_news(a, cfg, store, uni):
    from stockpred.data.gdelt import backfill
    from stockpred.serve.pipeline import recluster

    secs = uni.securities[: a.limit] if a.limit else uni.securities
    arts = asyncio.run(backfill(secs, a.days, timedelta(minutes=cfg.news_latency_minutes)))
    print(f"backfilled {store.upsert_articles(arts)} articles (headlines only; no body text)")
    recluster(store, cfg, hours=24 * a.days)


def cmd_extract(a, cfg, store, uni):
    from stockpred.nlp.extractor import Extractor
    from stockpred.nlp.providers import make_provider
    from stockpred.serve.pipeline import extract_pending

    async def run():
        prov = make_provider(cfg)
        ok, why = await prov.health()
        if not ok:
            print(f"LLM not ready: {why}")
            await prov.close()
            return
        ex = Extractor(prov, store, uni, temperature=cfg.llm_temperature, n_samples=cfg.llm_samples,
                       max_concurrent=cfg.llm_max_concurrent)
        total = 0
        while True:
            n = await extract_pending(store, ex, cfg)
            total += n
            if n == 0 or not a.all:
                break
        await prov.close()
        print(f"extracted {total} articles")

    asyncio.run(run())


def cmd_build_panel(a, cfg, store, uni):
    from stockpred.features.build import build_panel
    from stockpred.serve.pipeline import text_feature_frame

    px_dates = store.prices_df([cfg.benchmark]).date
    if px_dates.empty:
        sys.exit("no prices: run `ingest-prices` first")
    dates = pd.DatetimeIndex(sorted(px_dates.unique()))
    tf = None
    if a.text != "none":
        tf = text_feature_frame(store, uni, dates, a.text)
    panel = build_panel(store, uni, cfg.horizon_days, cfg.benchmark, text_features=tf,
                        use_factors=a.factors)
    panel.to_parquet(_panel_path(cfg))
    print(f"panel {panel.shape} -> {_panel_path(cfg)}")


def cmd_train(a, cfg, store, uni):
    from stockpred.models.train import TrainConfig, fit_bundle, run_experiment, save_bundle

    p = _panel_path(cfg)
    if not p.exists():
        sys.exit("no panel: run `build-panel` first")
    panel = pd.read_parquet(p)
    tc = TrainConfig(horizon=cfg.horizon_days, embargo=cfg.embargo_days, n_splits=a.splits,
                     test_days=a.test_days, use_lstm=a.lstm, tune_trials=a.tune, seed=cfg.random_seed)
    res = run_experiment(panel, tc, cfg.runs_dir)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols = ["ic_mean", "ic_t_hac", "ic_ir", "hit_rate", "hit_lo", "hit_hi", "q_spread", "n_dates"]
    print(res.metrics[cols].round(4).sort_values("ic_mean", ascending=False).to_string())
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.extras.items()})
    from stockpred.models.baselines import available_baselines

    sig = res.oof.assign(stack=res.stack)
    for k, v in available_baselines(panel).items():
        sig[f"baseline:{k}"] = v.reindex(sig.index)
    sig.to_parquet(cfg.artifacts_dir / "signals_oos.parquet")
    if not a.no_save:
        b = fit_bundle(panel, tc, res)
        print("saved", save_bundle(b, cfg.artifacts_dir / "models"))


def cmd_doctor(a, cfg, store, uni):
    from stockpred.nlp.providers import make_provider

    async def run():
        prov = make_provider(cfg)
        print("LLM:", cfg.llm_provider, cfg.llm_model, await prov.health())
        await prov.close()

    asyncio.run(run())
    px = store.prices_df()
    print(f"prices: {len(px)} rows, {px.symbol.nunique() if len(px) else 0} symbols")
    print(f"articles: {len(store.articles_df())}")
    print(f"extractions: {len(store.extraction_rows())} (article,ticker) rows")


def cmd_live(a, cfg, store, uni):
    asyncio.run(_live(a, cfg, store, uni))


async def _live(a, cfg, store, uni):
    import time

    from stockpred.models.train import load_bundle
    from stockpred.nlp.extractor import Extractor
    from stockpred.nlp.providers import make_provider
    from stockpred.serve import dashboard as dash
    from stockpred.serve.pipeline import (extract_pending, ingest_news_once, latest_rationales,
                                          log_signal_ledger, price_panel_cached, signals_table)

    prov = make_provider(cfg)
    llm_ok, detail = await prov.health()
    if not llm_ok and not a.force:
        dash.console.print(f"[red]LLM not ready:[/red] {detail}\n"
                           "Start it (e.g. `ollama serve`, `ollama pull %s`) or pass --force to run on stored data."
                           % cfg.llm_model)
        await prov.close()
        return
    ex = Extractor(prov, store, uni, temperature=cfg.llm_temperature, n_samples=cfg.llm_samples,
                   max_concurrent=cfg.llm_max_concurrent)
    try:
        bundle = load_bundle(cfg.artifacts_dir / "models")
        t = bundle.metrics.get("stack", {}).get("ic_t_hac")
        edge = ("NO SIGNIFICANT EDGE in walk-forward" if t is None or not t > 2.0
                else "walk-forward edge significant")
        info = f"stack(trained through {bundle.trained_through:%Y-%m-%d}; IC t={t:.1f}: {edge})" \
            if t is not None else f"stack(trained through {bundle.trained_through:%Y-%m-%d}; {edge})"
    except Exception:
        bundle, info = None, "none (run `train`); showing sentiment-only ranks"
    cache: dict = {}
    ui = None if (a.no_ui or a.once) else dash.Dashboard()
    if ui:
        ui.start()
    cycle, last = 0, 0.0
    try:
        while True:
            if time.time() - last >= cfg.news_interval_minutes * 60:
                cycle, last = cycle + 1, time.time()
                new = await ingest_news_once(store, uni, cfg)
                if llm_ok:
                    await extract_pending(store, ex, cfg)
                try:
                    pp = price_panel_cached(store, uni, cfg, cache)
                    sig = signals_table(store, uni, pp, bundle)
                    log_signal_ledger(store, sig, cfg.horizon_days, info)
                except Exception as exc:
                    log.exception("signal build failed")
                    dash.console.print(f"[red]signal build failed: {exc}[/red] (did you run ingest-prices?)")
                    if a.once:
                        break
                    sig = None
                if sig is not None:
                    st = dash.status_panel(cycle, cfg.llm_model, llm_ok, detail, info,
                                           cfg.news_interval_minutes * 60)
                    tb = dash.signals_table(sig, latest_rationales(store))
                    nw = dash.news_panel(new)
                    if ui:
                        ui.update(st, tb, nw)
                    else:
                        dash.console.print(st), dash.console.print(tb), dash.console.print(nw)
                if a.once:
                    break
            await asyncio.sleep(cfg.dashboard_refresh_s)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if ui:
            ui.stop()
        await prov.close()


def _bt_cfg(a):
    from stockpred.backtest.costs import CostModel
    from stockpred.backtest.engine import BacktestConfig

    return BacktestConfig(rebalance_days=a.rebalance, quantile=a.quantile, long_short=not a.long_only,
                          aum=a.aum, smooth=a.smooth,
                          cost=CostModel(a.commission_bps, a.spread_bps, a.impact, a.borrow_bps))


def cmd_backtest(a, cfg, store, uni):
    from stockpred.backtest.report import capacity_curve, decay_curve, run_suite, save_report
    from stockpred.data.macro import load_factors

    p = cfg.artifacts_dir / "signals_oos.parquet"
    if not p.exists():
        sys.exit("no out-of-sample signals: run `train` first")
    sigs = pd.read_parquet(p)
    prices = store.prices_df()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    bc = _bt_cfg(a)
    try:
        fac = load_factors(store)
    except Exception:
        fac = None
    suite = run_suite(sigs, prices, bc, benchmark=cfg.benchmark, factors=fac, runs_dir=cfg.runs_dir)
    cols = ["gross_sharpe", "net_sharpe", "sharpe_lo", "sharpe_hi", "ann_ret", "max_dd", "turnover_ann",
            "cost_drag_ann", "dsr", "alpha_ann", "alpha_t", "beta_mkt"]
    print(suite.table[cols].round(3).to_string())
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in suite.overfit.items()})
    print("DSR = P(true Sharpe > 0 after deflating for", suite.overfit.get("n_trials"), "trials); "
          "PBO = prob. the in-sample winner is below the OOS median.")
    if not suite.table.empty:
        best = suite.table.index[0]
        if a.curves:
            print(f"\ndecay (hold days), {best}:")
            print(decay_curve(sigs[best].dropna(), prices, bc).round(3).to_string())
            print(f"\ncapacity (AUM), {best}:")
            print(capacity_curve(sigs[best].dropna(), prices, bc).round(3).to_string())
    print("saved", save_report(suite, cfg.artifacts_dir / "backtest_report.json"))


def cmd_risk(a, cfg, store, uni):
    from stockpred.portfolio.paper import PaperAccount
    from stockpred.portfolio.risk import risk_report

    acct = PaperAccount(store)
    pos = acct.positions()
    if pos.empty:
        sys.exit("paper account is flat: nothing to report (run `paper-rebalance`)")
    px = store.prices_df(list(pos.index))
    rets = px.pivot(index="date", columns="symbol", values="adj_close").pct_change()
    nav = acct.nav(rets.index.max())
    last = px[px.date == px.date.max()].set_index("symbol")["close"]
    w = pos * last.reindex(pos.index) / nav
    import json

    print(json.dumps(risk_report(w, rets), indent=2, default=float))


def _runtime(a, cfg, store, uni):
    from stockpred.serve.jobs import Runtime

    rt = Runtime(cfg, store, uni, paper_enabled=True, paper_ignore_gate=getattr(a, "ignore_gate", False))
    rt.load_bundle()
    return rt


def cmd_paper(a, cfg, store, uni):
    from stockpred.portfolio.paper import PaperAccount
    from stockpred.serve.jobs import paper_cycle

    if a.action == "cycle":
        print(paper_cycle(_runtime(a, cfg, store, uni)))
        return
    acct = PaperAccount(store)
    h = acct.nav_history()
    print(f"cash {acct.cash:,.0f}  initial {acct.initial:,.0f}  nav rows {len(h)}")
    print(acct.positions().round(2).to_string() if len(acct.positions()) else "flat")
    if len(h):
        print(h.tail(10).round(4).to_string())
    print(acct.orders("pending").to_string(index=False) if a.action == "orders" else "")


def cmd_monitor(a, cfg, store, uni):
    import json

    from stockpred.serve.jobs import run_monitor
    from stockpred.serve.util import clean

    print(json.dumps(clean(run_monitor(_runtime(a, cfg, store, uni))), indent=2))


def cmd_serve(a, cfg, store, uni):
    import uvicorn

    from stockpred.serve.api import create_app
    from stockpred.serve.jobs import job_monitor, job_news, job_signals
    from stockpred.serve.scheduler import Scheduler

    rt = _runtime(a, cfg, store, uni)
    rt.paper_enabled, rt.auto_retrain = a.paper, a.auto_retrain
    sch = Scheduler()
    sch.add("news", lambda: job_news(rt), cfg.news_interval_minutes * 60)
    sch.add("signals", lambda: job_signals(rt), a.signal_minutes * 60)
    sch.add("monitor", lambda: job_monitor(rt), 6 * 3600, run_at_start=False)
    app = create_app(rt, sch)

    async def main():
        server = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="info"))
        task = asyncio.create_task(sch.run_forever())
        try:
            await server.serve()
        finally:
            sch.stop()
            await task

    asyncio.run(main())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockpred", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest-prices"); s.set_defaults(fn=cmd_ingest_prices)
    s.add_argument("--years", type=float, default=10); s.add_argument("--full", action="store_true")
    s.add_argument("--symbols", default=None, help="comma list; default = universe")

    sub.add_parser("ingest-macro").set_defaults(fn=cmd_ingest_macro)
    sub.add_parser("ingest-news").set_defaults(fn=cmd_ingest_news)

    s = sub.add_parser("backfill-news"); s.set_defaults(fn=cmd_backfill_news)
    s.add_argument("--days", type=int, default=60); s.add_argument("--limit", type=int, default=0)

    s = sub.add_parser("extract"); s.set_defaults(fn=cmd_extract)
    s.add_argument("--all", action="store_true", help="loop until no pending articles remain")

    s = sub.add_parser("build-panel"); s.set_defaults(fn=cmd_build_panel)
    s.add_argument("--text", choices=["llm", "lexicon", "none"], default="lexicon")
    s.add_argument("--factors", action="store_true", help="add Fama-French residual features (research only)")

    s = sub.add_parser("train"); s.set_defaults(fn=cmd_train)
    s.add_argument("--splits", type=int, default=6); s.add_argument("--test-days", type=int, default=126)
    s.add_argument("--lstm", action="store_true"); s.add_argument("--tune", type=int, default=0)
    s.add_argument("--no-save", action="store_true")

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)

    s = sub.add_parser("backtest"); s.set_defaults(fn=cmd_backtest)
    s.add_argument("--rebalance", type=int, default=5); s.add_argument("--quantile", type=float, default=0.2)
    s.add_argument("--long-only", action="store_true"); s.add_argument("--smooth", type=float, default=1.0)
    s.add_argument("--aum", type=float, default=1e6)
    s.add_argument("--commission-bps", type=float, default=0.5); s.add_argument("--spread-bps", type=float, default=2.0)
    s.add_argument("--impact", type=float, default=0.1); s.add_argument("--borrow-bps", type=float, default=30.0)
    s.add_argument("--curves", action="store_true", help="also print decay and capacity curves")

    sub.add_parser("risk").set_defaults(fn=cmd_risk)

    s = sub.add_parser("paper"); s.set_defaults(fn=cmd_paper)
    s.add_argument("action", choices=["status", "orders", "cycle"], nargs="?", default="status")
    s.add_argument("--ignore-gate", action="store_true",
                   help="trade the signal even without walk-forward evidence (measures the null)")

    sub.add_parser("monitor").set_defaults(fn=cmd_monitor)

    s = sub.add_parser("serve"); s.set_defaults(fn=cmd_serve)
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--paper", action="store_true"); s.add_argument("--auto-retrain", action="store_true")
    s.add_argument("--ignore-gate", action="store_true"); s.add_argument("--signal-minutes", type=int, default=60)

    s = sub.add_parser("live"); s.set_defaults(fn=cmd_live)
    s.add_argument("--once", action="store_true"); s.add_argument("--no-ui", action="store_true")
    s.add_argument("--force", action="store_true", help="run even if the LLM is unreachable")
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg = get_settings()
    store = _setup(cfg)
    try:
        a.fn(a, cfg, store, default_universe())
    finally:
        store.close()


if __name__ == "__main__":
    main()
