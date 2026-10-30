"""Glue: ingest -> dedup -> extract -> features -> model -> signals."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from stockpred.config.settings import Settings
from stockpred.config.universe import Universe
from stockpred.data.dedup import cluster_events
from stockpred.data.entities import EntityLinker
from stockpred.data.models import Article, utcnow
from stockpred.data.news_rss import RssIngestor, default_feeds
from stockpred.data.prices import latest_quotes
from stockpred.data.store import Store
from stockpred.features.build import build_panel
from stockpred.features.text import build_text_features, lexicon_rows
from stockpred.models.labels import cross_sectional_labels
from stockpred.models.train import Bundle, predict_bundle
from stockpred.nlp.extractor import Extractor
from stockpred.nlp.novelty import novelty_scores

log = logging.getLogger(__name__)


def recluster(store: Store, cfg: Settings, hours: float = 96) -> int:
    """(Re)assign event ids to recent articles; syndicated copies share one event_id."""
    df = store.articles_df(since=utcnow() - timedelta(hours=hours))
    if df.empty:
        return 0
    mapping = cluster_events(zip(df["id"], (df["title"] + ". " + df["summary"].fillna(""))),
                             threshold=cfg.dedup_threshold)
    store.set_event_ids(mapping)
    return len(set(mapping.values()))


async def ingest_news_once(store: Store, universe: Universe, cfg: Settings) -> list[Article]:
    ing = RssIngestor(default_feeds(universe), cfg.user_agent, store, cfg.max_article_age_hours)
    try:
        new = await ing.fetch_new()
    finally:
        await ing.close()
    n = store.upsert_articles(new)  # persist EVERYTHING: truncation happens only at extraction
    log.info("ingested %d new articles", n)
    recluster(store, cfg)
    return new


async def extract_pending(store: Store, extractor: Extractor, cfg: Settings) -> int:
    pending = store.articles_pending(limit=cfg.max_articles_per_cycle,
                                     since=utcnow() - timedelta(hours=cfg.max_article_age_hours))
    if not pending:
        return 0
    results = await extractor.extract_batch(pending)
    ok = sum(r.ok for r in results)
    log.info("extracted %d/%d articles", ok, len(pending))
    return ok


def text_feature_frame(store: Store, universe: Universe, dates: pd.DatetimeIndex, source: str = "llm",
                       model: str | None = None, clamp_to_last: bool = False) -> pd.DataFrame:
    """Per-(date,symbol) text features from stored extractions ('llm') or the lexicon ('lexicon')."""
    syms = universe.symbols
    arts = store.articles_df()
    if source == "lexicon":
        rows = lexicon_rows(arts, EntityLinker(universe)) if not arts.empty else pd.DataFrame()
        return build_text_features(rows, dates, syms, clamp_to_last=clamp_to_last)
    rows = store.extraction_rows(model)
    if rows.empty:
        return build_text_features(rows, dates, syms)
    a = arts.assign(text=arts["title"] + ". " + arts["summary"].fillna(""))
    nov = novelty_scores(a[["id", "text", "available_at"]])
    rows = rows.assign(event_id=rows["event_id"].fillna(rows["article_id"]))
    return build_text_features(rows, dates, syms, novelty=nov, clamp_to_last=clamp_to_last)


def live_rows(price_panel: pd.DataFrame, store: Store, universe: Universe, source: str = "llm") -> pd.DataFrame:
    """Feature vector as of *now*: price features from the last close + text through the latest news."""
    dates = price_panel.index.get_level_values("date").unique().sort_values()
    nxt = dates[-1] + pd.offsets.BDay(1)
    ext = dates.append(pd.DatetimeIndex([nxt]))
    tf = text_feature_frame(store, universe, ext, source, clamp_to_last=True)
    last = price_panel.xs(dates[-1], level="date").copy()
    last["date"] = nxt
    live = last.reset_index().set_index(["date", "symbol"])
    drop = [c for c in live.columns if c in tf.columns]
    live = live.drop(columns=drop).join(tf, how="left")
    return live


def signals_table(
    store: Store, universe: Universe, price_panel: pd.DataFrame, bundle: Bundle | None,
    source: str = "llm",
) -> pd.DataFrame:
    live = live_rows(price_panel, store, universe, source)
    quotes = latest_quotes(store, universe.symbols)
    if bundle is not None:
        for c in bundle.features:
            if c not in live.columns:
                live[c] = np.nan
        pred = predict_bundle(bundle, live)
        out = pred.reset_index(level="date", drop=True)
        out["model"] = "stack"
    else:
        s = live["sent_ew1d"].reset_index(level="date", drop=True) if "sent_ew1d" in live else pd.Series(0.0, index=universe.symbols)
        out = pd.DataFrame({"score": s})
        out["p_up"] = np.nan
        out["exp_excess_ret"] = np.nan
        out["lo"] = out["hi"] = np.nan
        out["label"] = cross_sectional_labels(out["score"])
        out["model"] = "UNTRAINED: sentiment only"
    lv = live.reset_index(level="date", drop=True)
    for c in ("sent_ew1d", "n_events_5d", "buzz_z"):
        out[c] = lv[c] if c in lv else np.nan
    out["price"] = [quotes.get(s, {}).get("price", np.nan) for s in out.index]
    out["chg_pct"] = [quotes.get(s, {}).get("change_pct", np.nan) for s in out.index]
    return out


def latest_rationales(store: Store, hours: float = 48) -> dict[str, tuple[str, float]]:
    """symbol -> (rationale, confidence) of the most confident recent extraction."""
    rows = store.extraction_rows()
    if rows.empty:
        return {}
    rows = rows[rows["available_at"] >= utcnow() - timedelta(hours=hours)]
    out = {}
    for sym, g in rows.groupby("ticker"):
        r = g.sort_values("confidence", ascending=False).iloc[0]
        out[sym] = (r["rationale"], float(r["confidence"]))
    return out


def log_signal_ledger(store: Store, sig: pd.DataFrame, horizon: int, model_name: str) -> None:
    now = utcnow()
    rows = [(now, sym, model_name, horizon, float(r.score),
             None if pd.isna(r.p_up) else float(r.p_up),
             None if pd.isna(r.lo) else float(r.lo), None if pd.isna(r.hi) else float(r.hi), r.label)
            for sym, r in sig.iterrows()]
    store.log_predictions(rows)


def price_panel_cached(store: Store, universe: Universe, cfg: Settings, cache: dict) -> pd.DataFrame:
    """Rebuild the (expensive) price panel only when new daily bars arrive."""
    last = store.prices_df([cfg.benchmark]).date.max()
    if cache.get("last") != last:
        cache["panel"] = build_panel(store, universe, cfg.horizon_days, cfg.benchmark)
        cache["last"] = last
    return cache["panel"]
