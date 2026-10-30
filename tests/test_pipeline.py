from datetime import timedelta

import pandas as pd

from conftest import make_prices
from stockpred.cli import build_parser
from stockpred.config.settings import Settings
from stockpred.data.models import make_article, utcnow
from stockpred.data.store import Store
from stockpred.features.build import build_panel
from stockpred.nlp.extractor import Extractor
from stockpred.serve import dashboard as dash
from stockpred.serve.pipeline import (extract_pending, live_rows, recluster, signals_table,
                                      text_feature_frame)
from test_nlp import FakeLLM, _reply


def _store(universe):
    st = Store()
    frames = []
    for i, s in enumerate(universe.symbols + ["SPY"]):
        d = make_prices(n=700, seed=20 + i, start="2023-06-01").reset_index(names="date")
        d["symbol"] = s
        frames.append(d)
    st.upsert_prices(pd.concat(frames))
    return st


def test_settings_defaults_and_env(monkeypatch):
    assert Settings().horizon_days == 5
    monkeypatch.setenv("STOCKPRED_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("STOCKPRED_HORIZON_DAYS", "10")
    s = Settings()
    assert s.llm_provider == "anthropic" and s.horizon_days == 10


def test_settings_validation():
    import pytest

    with pytest.raises(ValueError):
        Settings(horizon_days=0)


def test_cli_parser_commands():
    p = build_parser()
    for argv in (["ingest-prices", "--years", "2"], ["train", "--lstm", "--tune", "3"],
                 ["live", "--once", "--force"], ["build-panel", "--text", "lexicon"], ["doctor"]):
        assert p.parse_args(argv).cmd == argv[0]


async def test_extract_pending_then_signals_without_model(universe):
    st = _store(universe)
    now = utcnow()
    arts = [make_article(f"Apple raises guidance, iPhone demand strong {i}", "", f"http://x/{i}", "wire",
                         now - timedelta(hours=i + 1), ingested_at=now - timedelta(hours=i)) for i in range(3)]
    st.upsert_articles(arts)
    assert recluster(st, Settings()) >= 1
    llm = FakeLLM([_reply(sent=0.8)])
    ok = await extract_pending(st, Extractor(llm, st, universe), Settings())
    assert ok == 3
    assert not st.articles_pending()  # nothing left, and nothing was silently dropped

    panel = build_panel(st, universe, 5, "SPY", garch=False, regime=False)
    live = live_rows(panel, st, universe)
    nxt = panel.index.get_level_values("date").max() + pd.offsets.BDay(1)
    assert live.index.get_level_values("date").unique().tolist() == [nxt]
    assert live.loc[(nxt, "AAPL"), "n_events_5d"] >= 1  # news since last close reaches the live row
    assert live.loc[(nxt, "MSFT"), "n_events_5d"] == 0

    sig = signals_table(st, universe, panel, bundle=None)
    assert sig.loc["AAPL", "sent_ew1d"] > 0 and "UNTRAINED" in sig["model"].iloc[0]
    assert sig["label"].isin(["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]).all()
    # dashboard renders without error on a sparse frame
    tbl = dash.signals_table(sig, {})
    assert tbl.row_count == len(universe)
    st.close()


def test_text_feature_frame_lexicon_source(universe):
    st = _store(universe)
    now = utcnow()
    st.upsert_articles([make_article("Exxon faces lawsuit and fraud probe, shares plunge", "", "http://e/1", "w",
                                     now - timedelta(hours=3), ingested_at=now - timedelta(hours=2))])
    dates = pd.DatetimeIndex(sorted(st.prices_df(["SPY"]).date.unique()))
    dates = dates.append(pd.DatetimeIndex([dates[-1] + pd.offsets.BDay(1)]))
    tf = text_feature_frame(st, universe, dates, source="lexicon", clamp_to_last=True)
    assert tf.xs("XOM", level="symbol")["sent_ew5d"].iloc[-1] < 0
    st.close()


def test_score_bar_symmetry():
    a, b = dash.score_bar(0.5), dash.score_bar(-0.5)
    assert len(a.plain) == len(b.plain) == 17  # same width either sign
    assert a.plain.index("│") == b.plain.index("│") == 8  # shared zero axis
    assert dash.score_bar(float("nan")).plain.count("·") == 17
