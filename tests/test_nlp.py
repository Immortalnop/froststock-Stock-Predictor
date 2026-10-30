import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stockpred.data.models import make_article
from stockpred.data.store import Store
from stockpred.nlp.calibration import ConfidenceCalibrator, benchmark_scorers, evaluate_classifier
from stockpred.nlp.extractor import Extractor, cache_key, mask_entities
from stockpred.nlp.lexicon import score_text
from stockpred.nlp.novelty import novelty_scores
from stockpred.nlp.schema import extract_json, parse_output


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert extract_json("nothing") == {}


def test_schema_clips_and_coerces():
    out = parse_output(json.dumps({
        "tickers": [{"ticker": "aapl", "sentiment": 7, "relevance": "x"}],
        "event_type": "weird", "horizon": "??", "confidence": None}))
    t = out.tickers[0]
    assert t.ticker == "AAPL" and t.sentiment == 1.0 and t.relevance == 1.0
    assert out.event_type == "other" and out.horizon == "short"
    assert out.confidence == 0.0  # missing confidence must not default to a flattering value


def test_schema_accepts_legacy_dict_format():
    out = parse_output('{"tickers": {"AAPL": 0.4}, "confidence": 0.9}')
    assert out.tickers[0].sentiment == 0.4


def test_parse_failure_raises():
    with pytest.raises(ValueError):
        parse_output("garbage")


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def generate(self, prompt, *, temperature):
        self.calls += 1
        r = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    async def health(self):
        return True, "ok"

    async def close(self):
        pass


def _art(i, text="Apple beats earnings estimates on strong iPhone demand"):
    return make_article(text, "", f"http://x/{i}", "s", datetime(2026, 1, 5, 12), latency=pd.Timedelta(0))


def _reply(sent=0.8, conf=0.9, tickers=("AAPL",)):
    return json.dumps({"tickers": [{"ticker": t, "sentiment": sent, "relevance": 1} for t in tickers],
                       "event_type": "earnings", "horizon": "short", "confidence": conf, "rationale": "r"})


async def test_extractor_caches_and_does_not_cache_failures(universe):
    st = Store()
    llm = FakeLLM([RuntimeError("boom"), _reply()])
    ex = Extractor(llm, st, universe)
    a = _art(1)
    st.upsert_articles([a])
    r1 = await ex.extract(a)
    assert not r1.ok and st.get_cached_extraction(r1.cache_key) is None  # failure not cached
    r2 = await ex.extract(a)
    assert r2.ok and r2.tickers[0].ticker == "AAPL" and llm.calls == 2
    # same text, different article id -> cache hit, no new LLM call, correct article_id
    b = _art(2)
    st.upsert_articles([b])
    r3 = await ex.extract(b)
    assert llm.calls == 2 and r3.article_id == b.id
    assert len(st.extraction_rows()) == 2


async def test_extractor_rejects_hallucinated_ticker(universe):
    st = Store()
    llm = FakeLLM([_reply(tickers=("AAPL", "XOM"))])  # text only mentions Apple
    a = _art(1)
    st.upsert_articles([a])
    r = await Extractor(llm, st, universe).extract(a)
    assert [t.ticker for t in r.tickers] == ["AAPL"]


async def test_self_consistency_agreement(universe):
    st = Store()
    llm = FakeLLM([_reply(sent=0.9), _reply(sent=-0.9), _reply(sent=0.9)])
    a = _art(1)
    st.upsert_articles([a])
    r = await Extractor(llm, st, universe, n_samples=3).extract(a)
    assert r.n_samples == 3 and r.agreement < 0.8 and r.confidence < 0.9


def test_cache_key_depends_on_model_and_text(universe):
    k = cache_key("m1", "text", universe)
    assert k != cache_key("m2", "text", universe) and k != cache_key("m1", "text2", universe)


def test_mask_entities(universe):
    out = mask_entities("Apple's iPhone sales and MSFT Azure", universe)
    assert "Apple" not in out and "iPhone" not in out and "MSFT" not in out


def test_lexicon_negation():
    assert score_text("profits surge, strong growth") > 0.3
    assert score_text("lawsuit and fraud probe, shares plunge") < -0.3
    assert score_text("no growth") < score_text("growth")


def test_novelty_decreases_for_repeats():
    t0 = pd.Timestamp("2026-01-05 10:00")
    df = pd.DataFrame({
        "id": ["a", "b", "c"],
        "text": ["Apple earnings beat expectations strongly", "Apple earnings beat expectations strongly today",
                 "Oil prices fall on supply glut"],
        "available_at": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)]})
    n = novelty_scores(df)
    assert n["a"] == 1.0 and n["b"] < 0.3 and n["c"] > 0.8


def test_confidence_calibrator_learns_monotone_map():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 4000)
    sent = rng.choice([-1.0, 1.0], 4000)
    correct = rng.uniform(size=4000) < (0.5 + 0.4 * conf)
    ret = np.where(correct, sent, -sent) * 0.01
    cal = ConfidenceCalibrator().fit(conf, sent, ret)
    p = cal.predict([0.05, 0.95])
    assert p[1] > p[0] + 0.2


def test_benchmark_scorers():
    df = pd.DataFrame({"text": ["profits surge strong growth", "lawsuit fraud plunge", "the meeting is tuesday"],
                       "label": ["positive", "negative", "neutral"]})
    res = benchmark_scorers(df, {"lex": score_text})
    assert res.loc[0, "accuracy"] == 1.0
    assert evaluate_classifier(["positive"], ["positive"])["n"] == 1
