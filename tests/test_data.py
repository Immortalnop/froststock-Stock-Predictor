from datetime import datetime, timedelta

import pandas as pd

from stockpred.data.dedup import cluster_events
from stockpred.data.entities import EntityLinker
from stockpred.data.macro import parse_fred_csv, parse_french_csv
from stockpred.data.models import clean_text, make_article
from stockpred.data.news_rss import Feed, parse_feed
from stockpred.data.prices import normalize_yf
from stockpred.data.store import Store
from stockpred.data.universe_sp500 import reconstruct_membership

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>Apple beats earnings</title><link>http://a/1</link>
<description>&lt;p&gt;Strong &amp;amp; fast&lt;/p&gt;</description>
<pubDate>Mon, 05 Jan 2026 15:00:00 GMT</pubDate></item>
<item><title>No date item</title><link>http://a/2</link></item>
<item><title>Future item</title><link>http://a/3</link><pubDate>Mon, 05 Jan 2026 23:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_clean_text_strips_html_and_entities():
    assert clean_text("<p>A &amp; B</p>\n  x") == "A & B x"


def test_parse_feed_uses_utc_and_drops_undated():
    now = datetime(2026, 1, 5, 16, 0)
    arts, skipped = parse_feed(RSS, Feed("http://f"), now=now, ingested_at=now)
    assert [a.title for a in arts] == ["Apple beats earnings"]
    assert arts[0].published_at == datetime(2026, 1, 5, 15, 0)  # not shifted by local tz
    assert skipped == 2  # undated + future-dated
    assert arts[0].summary == "Strong & fast"


def test_available_at_backfill_vs_live():
    pub = datetime(2026, 1, 5, 12, 0)
    live = make_article("t", "", "u", "s", pub, ingested_at=datetime(2026, 1, 5, 12, 5))
    assert live.available_at == datetime(2026, 1, 5, 12, 5)
    bf = make_article("t", "", "u", "s", pub, latency=timedelta(minutes=15))
    assert bf.available_at == pub + timedelta(minutes=15)


def test_store_roundtrip_and_dedupe(tmp_path):
    st = Store(tmp_path / "t.duckdb")
    a = make_article("Hello", "w", "u1", "s", datetime(2026, 1, 1, 10))
    assert st.upsert_articles([a]) == 1
    assert st.upsert_articles([a]) == 0
    assert st.known_article_ids([a.id, "zzz"]) == {a.id}
    assert len(st.articles_pending()) == 1
    st.close()


def test_store_prices_and_series():
    st = Store()
    df = pd.DataFrame({"symbol": ["A", "A"], "date": ["2026-01-02", "2026-01-05"], "open": [1, 2.0],
                       "high": [2, 3.0], "low": [1, 1.0], "close": [1.5, 2.5], "adj_close": [1.5, 2.5],
                       "volume": [10, 20.0]})
    st.upsert_prices(df)
    st.upsert_prices(df)  # idempotent
    assert len(st.prices_df(["A"])) == 2
    assert st.last_price_date("A") == pd.Timestamp("2026-01-05")
    st.upsert_series("vix", pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-02", "2026-01-05"])))
    assert st.series("vix").iloc[-1] == 2.0


def test_dedup_clusters_syndicated_copies():
    items = [
        ("a", "Apple reports record quarterly revenue driven by strong iPhone sales in China market"),
        ("b", "Apple reports record quarterly revenue driven by strong iPhone sales in China markets"),
        ("c", "Federal Reserve holds interest rates steady citing inflation progress and labor data"),
    ]
    m = cluster_events(items, threshold=0.5)
    assert m["a"] == m["b"]
    assert m["c"] != m["a"]


def test_entity_linking_and_verification(universe):
    lk = EntityLinker(universe)
    assert lk.link("The iPhone maker posted gains") == {"AAPL"}
    assert lk.link("$MSFT jumps; Exxon slides") == {"MSFT", "XOM"}
    assert lk.link("Visa card volumes") == {"V"}
    assert lk.link("Was V for victory") == set()  # ambiguous 1-letter symbol needs cashtag/name
    assert lk.verify(["AAPL", "XOM"], "Apple unveils new Mac") == ["AAPL"]


def test_parsers_fred_and_french():
    s = parse_fred_csv("observation_date,DGS10\n2026-01-02,4.1\n2026-01-03,.\n2026-01-06,4.2\n")
    assert list(s.values) == [4.1, 4.2]
    txt = ("junk\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20260102,1.0,0.5,-0.5,0.1,0.2,0.01\n"
           "20260105,-1.0,0.0,0.5,0.1,0.2,0.01\n\nAnnual\n2025,1,1,1,1,1,1\n")
    ff = parse_french_csv(txt)
    assert ff.shape == (2, 6) and abs(ff["Mkt-RF"].iloc[0] - 0.01) < 1e-12


def test_normalize_yf_multiindex():
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    cols = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    raw = pd.DataFrame([[1, 2, 1, 1.5, 1.4, 10], [2, 3, 1, 2.5, 2.4, 20]], index=idx, columns=cols)
    out = normalize_yf(raw, ["AAA"])
    assert list(out["symbol"].unique()) == ["AAA"] and len(out) == 2
    assert out["adj_close"].iloc[0] == 1.4


def test_sp500_membership_rollback():
    current = {"A", "B", "NEW"}
    ch = pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2022-01-01"]),
                       "added": ["NEW", "B"], "removed": ["OLD", "GONE"]})
    assert reconstruct_membership(current, ch, "2025-01-01") == {"A", "B", "NEW"}
    assert reconstruct_membership(current, ch, "2023-01-01") == {"A", "B", "OLD"}
    assert reconstruct_membership(current, ch, "2021-01-01") == {"A", "OLD", "GONE"}
