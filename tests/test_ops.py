import asyncio

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import make_prices
from stockpred.config.settings import Settings
from stockpred.data.store import Store
from stockpred.mlops.drift import RetrainPolicy, drift_report, psi, retrain_decision
from stockpred.serve.api import create_app
from stockpred.serve.jobs import Runtime
from stockpred.serve.ledger import entry_dates, live_performance, realized_ledger
from stockpred.serve.scheduler import Scheduler


def test_psi_and_ks_detect_shift_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 2000)) < 0.1
    assert psi(a, rng.normal(1.0, 1, 2000)) > 0.25
    ref = pd.DataFrame({"x": a[:3000], "y": rng.normal(size=3000)})
    cur = pd.DataFrame({"x": rng.normal(1.5, 1, 500), "y": rng.normal(size=500)})
    rep = drift_report(ref, cur, ["x", "y"])
    assert bool(rep.loc["x", "drift"]) and not bool(rep.loc["y", "drift"]) and rep.index[0] == "x"


def test_retrain_policy():
    none = pd.DataFrame({"drift": [False, False]})
    assert retrain_decision(none, 5, 0.5, 100) == (False, [])
    ok, why = retrain_decision(none, 90, None, 0)
    assert ok and "age" in why[0]
    ok, why = retrain_decision(none, 5, -3.0, 100)
    assert ok and "live IC" in why[0]
    assert not retrain_decision(none, 5, -3.0, 10)[0]  # too few live observations to act on
    many = pd.DataFrame({"drift": [True, True, False]})
    assert retrain_decision(many, 1, 0, 100, RetrainPolicy(max_drift_share=0.5))[0]


def test_entry_dates_follow_the_next_open():
    days = pd.bdate_range("2024-03-04", periods=10)  # Mon..
    # 15:00 UTC = 10:00 EST (after open) on Tue -> fills Wed; 12:00 UTC = 07:00 EST (before open) on Tue -> fills Tue
    # Saturday -> Monday
    m = pd.Series(pd.to_datetime(["2024-03-05 15:00", "2024-03-05 12:00", "2024-03-09 15:00"]))
    e = entry_dates(m, days)
    assert list(e) == [pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-11")]


def _store_with_predictions(horizon=2, informative=True):
    st = Store()
    syms = [f"S{i}" for i in range(10)] + ["SPY"]
    frames = []
    for i, s in enumerate(syms):
        d = make_prices(n=120, seed=40 + i, start="2024-01-02", drift=0.0).reset_index(names="date")
        d["symbol"] = s
        frames.append(d)
    px = pd.concat(frames)
    st.upsert_prices(px)
    op = px.pivot(index="date", columns="symbol", values="open")
    rows = []
    days = op.index
    for t in range(10, 100):
        entry = days[t]
        fut = op.iloc[t + horizon] / op.iloc[t] - 1
        bench = fut["SPY"]
        # 3h before midnight UTC of the entry day: after the prior close, so the order fills at `entry`
        made_utc = pd.Timestamp(entry) - pd.Timedelta(hours=3)
        for s in syms[:-1]:
            score = (fut[s] - bench) if informative else np.random.default_rng([t, int(s[1:])]).normal()
            rows.append((made_utc.to_pydatetime(), s, "stack", horizon, float(score), 0.5, -0.01, 0.01, "NEUTRAL"))
    st.log_predictions(rows)
    return st


def test_ledger_ic_is_one_for_oracle_and_zero_for_noise():
    st = _store_with_predictions(informative=True)
    led = realized_ledger(st, 2, "SPY")
    assert led["realized"].notna().sum() > 500
    perf = live_performance(led, 2)
    assert perf["ic_mean"] > 0.95 and perf["hit_rate"] > 0.95
    st.close()
    st = _store_with_predictions(informative=False)
    perf = live_performance(realized_ledger(st, 2, "SPY"), 2)
    assert abs(perf["ic_mean"]) < 0.1
    st.close()


def test_ledger_unrealised_rows_are_nan_not_dropped():
    st = _store_with_predictions()
    # a prediction whose exit bar does not exist yet
    last = st.prices_df(["SPY"]).date.max()
    st.log_predictions([((last).to_pydatetime(), "S0", "stack", 2, 0.1, 0.5, 0, 0, "NEUTRAL")])
    led = realized_ledger(st, 2, "SPY")
    assert led["realized"].isna().any()
    st.close()


@pytest.fixture
def client(tmp_path):
    st = _store_with_predictions()
    cfg = Settings(artifacts_dir=tmp_path / "art", data_dir=tmp_path / "d", runs_dir=tmp_path / "r", horizon_days=2)
    rt = Runtime(cfg, st, None)
    sch = Scheduler()
    sch.add("noop", lambda: asyncio.sleep(0), 60, run_at_start=False)
    yield TestClient(create_app(rt, sch))
    st.close()


def test_api_endpoints(client):
    h = client.get("/health").json()
    assert h["ok"] and h["model_loaded"] is False and "not financial advice" in h["disclaimer"]
    s = client.get("/signals").json()
    assert len(s["signals"]) == 10 and s["signals"][0]["score"] >= s["signals"][-1]["score"]
    assert client.get("/signals/s3").status_code == 200
    assert client.get("/signals/NOPE").status_code == 404
    assert client.get("/model").status_code == 404
    assert client.get("/monitor").status_code == 404
    assert client.get("/performance").json()["n_realized"] > 0
    assert client.get("/paper/nav").json()["initial"] == 1_000_000
    assert client.get("/jobs").json()[0]["name"] == "noop"


async def test_scheduler_isolates_failures_and_does_not_overlap():
    sch = Scheduler()
    calls = {"ok": 0, "slow": 0}

    async def bad():
        raise RuntimeError("boom")

    async def ok():
        calls["ok"] += 1

    async def slow():
        calls["slow"] += 1
        await asyncio.sleep(0.5)

    sch.add("bad", bad, 0.05)
    sch.add("ok", ok, 0.05)
    sch.add("slow", slow, 0.05)
    task = asyncio.create_task(sch.run_forever(tick_s=0.02))
    await asyncio.sleep(0.4)
    sch.stop()
    await task
    st = {j["name"]: j for j in sch.status()}
    assert st["bad"]["last_ok"] is False and "boom" in st["bad"]["last_error"]
    assert calls["ok"] >= 3 and st["ok"]["last_ok"] is True  # the failing job did not stop the others
    assert calls["slow"] == 1  # still running: never launched a second copy


def test_clean_makes_nan_json_safe():
    import json

    from stockpred.serve.util import clean

    out = clean({"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3), "d": [np.nan, 1.0], "e": np.True_})
    assert out == {"a": None, "b": None, "c": 3, "d": [None, 1.0], "e": True}
    json.dumps(out, allow_nan=False)
