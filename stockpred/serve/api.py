"""Read-only FastAPI service over the store, ledger, monitor output and paper account.

It must share a process with the writer (DuckDB allows one read-write process): `stockpred serve`
runs the scheduler and this API together.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException

from stockpred.serve.jobs import Runtime, monitor_path
from stockpred.serve.ledger import live_performance, predictions_df, realized_ledger
from stockpred.serve.scheduler import Scheduler
from stockpred.serve.util import clean

DISCLAIMER = "Research output: statistical estimates, not financial advice."


def create_app(rt: Runtime, scheduler: Scheduler | None = None) -> FastAPI:
    app = FastAPI(title="stockpred", version="0.2.0", description=DISCLAIMER)

    @app.get("/health")
    def health():
        px = rt.store.prices_df([rt.cfg.benchmark])
        last = None if px.empty else px.date.max()
        b = rt.bundle or rt.load_bundle()
        age = None if b is None else (pd.Timestamp.now() - b.trained_through).days
        return clean({"ok": True, "last_price_date": last, "model_loaded": b is not None, "model_age_days": age,
                       "n_articles": len(rt.store.articles_df()), "disclaimer": DISCLAIMER})

    @app.get("/model")
    def model():
        p = rt.cfg.artifacts_dir / "models" / "latest" / "meta.json"
        if not p.exists():
            raise HTTPException(404, "no trained model: run `stockpred train`")
        meta = json.loads(p.read_text())
        t = meta.get("metrics", {}).get("stack", {}).get("ic_t_hac")
        meta["walk_forward_edge_significant"] = bool(t is not None and t > 2.0)
        return clean(meta)

    @app.get("/signals")
    def signals():
        pr = predictions_df(rt.store)
        if pr.empty:
            return {"made_at": None, "signals": [], "disclaimer": DISCLAIMER}
        last = pr[pr.made_at == pr.made_at.max()].sort_values("score", ascending=False)
        b = rt.bundle or rt.load_bundle()
        t = None if b is None else b.metrics.get("stack", {}).get("ic_t_hac")
        return clean({"made_at": last.made_at.iloc[0], "model": last.model.iloc[0],
                      "walk_forward_ic_t": t,
                      "warning": None if (t is not None and t > 2.0) else
                      "No statistically significant walk-forward edge: labels are cross-sectional ranks, not advice.",
                       "signals": last[["symbol", "score", "p_up", "lo", "hi", "label"]].to_dict("records"),
                       "disclaimer": DISCLAIMER})

    @app.get("/signals/{symbol}")
    def signal_history(symbol: str, limit: int = 100):
        pr = predictions_df(rt.store)
        h = pr[pr.symbol == symbol.upper()].tail(max(1, min(limit, 1000)))
        if h.empty:
            raise HTTPException(404, f"no predictions for {symbol}")
        return clean(h[["made_at", "score", "p_up", "lo", "hi", "label"]].to_dict("records"))

    @app.get("/performance")
    def performance():
        led = realized_ledger(rt.store, rt.cfg.horizon_days, rt.cfg.benchmark)
        return clean({"horizon_days": rt.cfg.horizon_days, **live_performance(led, rt.cfg.horizon_days)})

    @app.get("/monitor")
    def monitor():
        p = monitor_path(rt.cfg)
        if not p.exists():
            raise HTTPException(404, "no monitor report yet")
        return json.loads(p.read_text())

    @app.get("/paper/nav")
    def paper_nav():
        from stockpred.portfolio.paper import PaperAccount

        h = PaperAccount(rt.store).nav_history()
        return clean({"initial": PaperAccount(rt.store).initial, "nav": [{"date": d, **r}
                                                                         for d, r in h.to_dict("index").items()]})

    @app.get("/paper/positions")
    def paper_positions():
        from stockpred.portfolio.paper import PaperAccount

        return clean(PaperAccount(rt.store).positions().to_dict())

    @app.get("/jobs")
    def jobs():
        return scheduler.status() if scheduler else []

    return app
