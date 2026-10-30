import numpy as np
import pandas as pd

from conftest import make_prices
from stockpred.features.build import build_panel, forward_returns
from stockpred.features.cross import cs_rank, factor_exposures, rolling_beta
from stockpred.features.fracdiff import ffd_weights, frac_diff_ffd, min_d_stationary
from stockpred.features.technical import technical_features
from stockpred.features.text import assign_feature_date, build_text_features
from stockpred.features.volatility import garch_vol_forecast, volatility_features
from stockpred.models.targets import meta_labels, triple_barrier


def _causal(fn, px, ks=(300, 450, 600)):
    """Feature at row k-1 must not change when all later data is removed."""
    full = fn(px)
    for k in ks:
        part = fn(px.iloc[:k])
        a, b = part.iloc[-1], full.iloc[k - 1]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9, atol=1e-12)


def test_technical_features_are_causal(px):
    _causal(technical_features, px)


def test_volatility_features_are_causal(px):
    _causal(lambda p: volatility_features(p, garch=False), px)


def test_garch_forecast_is_causal_and_positive():
    px = make_prices(n=700, seed=3)
    lr = np.log(px.close).diff()
    full = garch_vol_forecast(lr, refit_every=63, min_obs=504)
    part = garch_vol_forecast(lr.iloc[:640], refit_every=63, min_obs=504)
    both = pd.concat([full.iloc[:640], part], axis=1).dropna()
    assert len(both) > 50
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-8)
    v = full.dropna()
    assert (v > 0.02).all() and (v < 2.0).all()  # sane annualised vol for 1.5% daily noise


def test_forward_return_is_next_open_to_open(px):
    f = forward_returns(px, 5)
    t = 100
    assert np.isclose(f.iloc[t], px.open.iloc[t + 6] / px.open.iloc[t + 1] - 1)
    assert f.iloc[-6:].isna().all()


def test_rsi_bounds(px):
    r = technical_features(px)["rsi_14"].dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_adjustment_removes_split_gap():
    p = make_prices(n=300, seed=1)
    p.loc[p.index[150]:, ["open", "high", "low", "close", "adj_close"]] *= 1.0
    q = p.copy()
    # 2:1 split at row 150: raw prices halve, adj_close history is restated (halved before the split)
    q.loc[q.index[150]:, ["open", "high", "low", "close"]] /= 2
    q.loc[q.index[150]:, "adj_close"] = q.loc[q.index[150]:, "close"]
    q.loc[:q.index[149], "adj_close"] = q.loc[:q.index[149], "close"] / 2
    g = technical_features(q)["gap"].iloc[150]
    assert abs(g) < 0.05


def test_assign_feature_date_cutoffs():
    dates = pd.bdate_range("2026-01-05", periods=6)  # Mon..Mon
    # UTC times: 20:30Z = 15:30 ET (Mon, in view), 21:30Z = 16:30 ET (after close -> Tue)
    s = pd.Series(pd.to_datetime(["2026-01-05 20:30", "2026-01-05 21:30", "2026-01-09 22:00",
                                  "2026-01-10 15:00", "2026-01-05 21:00"]))
    out = assign_feature_date(s, dates)
    assert list(out.iloc[:4]) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06"),
                                  pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-12")]
    assert out.iloc[4] == pd.Timestamp("2026-01-05")  # exactly 16:00 ET is still in view


def test_assign_feature_date_beyond_range_is_nat():
    dates = pd.bdate_range("2026-01-05", periods=2)
    out = assign_feature_date(pd.Series(pd.to_datetime(["2026-01-08 15:00"])), dates)
    assert out.isna().all()


def _rows(entries):
    return pd.DataFrame(entries, columns=["article_id", "event_id", "available_at", "ticker", "sentiment",
                                          "relevance", "confidence", "event_type"])


def test_text_features_event_dedup_and_decay():
    dates = pd.bdate_range("2026-01-05", periods=8)
    t = pd.Timestamp("2026-01-05 15:00")  # 10:00 ET Monday
    rows = _rows([("a1", "E1", t, "AAPL", 0.8, 1, 1, "earnings"),
                  ("a2", "E1", t + pd.Timedelta(minutes=5), "AAPL", 0.8, 1, 1, "earnings"),  # syndicated copy
                  ("b1", "E2", t, "MSFT", -0.5, 1, 1, "other")])
    f = build_text_features(rows, dates, ["AAPL", "MSFT"])
    mon = f.xs(dates[0], level="date")
    assert mon.loc["AAPL", "n_events_1d"] == 1.0  # two articles, one event
    assert mon.loc["AAPL", "sent_ew1d"] > 0.7 and mon.loc["MSFT", "sent_ew1d"] < -0.4
    assert mon.loc["AAPL", "ev_earnings_3d"] == 1.0
    # mass halves each halflife-1 day, average sentiment stays ~constant with no new news
    m0, m1 = f.loc[(dates[0], "AAPL"), "sent_mass1d"], f.loc[(dates[1], "AAPL"), "sent_mass1d"]
    assert np.isclose(m1, m0 * 0.5)
    assert f.loc[(dates[3], "AAPL"), "days_since_news"] == 3.0


def test_text_features_do_not_leak_future():
    dates = pd.bdate_range("2026-01-05", periods=5)
    late = pd.Timestamp("2026-01-07 15:00")  # Wed 10:00 ET
    f = build_text_features(_rows([("a", "E", late, "AAPL", 0.9, 1, 1, "other")]), dates, ["AAPL"])
    assert (f.xs("AAPL", level="symbol").loc[:dates[1], "n_events_1d"] == 0).all()
    assert f.loc[(dates[2], "AAPL"), "n_events_1d"] == 1.0


def test_text_features_empty_input():
    f = build_text_features(pd.DataFrame(), pd.bdate_range("2026-01-05", periods=3), ["AAPL"])
    assert (f["n_events_5d"] == 0).all() and len(f) == 3


def test_fracdiff_limits():
    assert np.allclose(ffd_weights(0.0), [1.0])
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=500)), name="x")
    d1 = frac_diff_ffd(s, 1.0)
    np.testing.assert_allclose(d1.iloc[10:], s.diff().iloc[10:], atol=1e-9)
    d = min_d_stationary(s)
    assert 0.1 <= d <= 1.0


def test_fracdiff_is_causal():
    s = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=400)))
    a, b = frac_diff_ffd(s, 0.4), frac_diff_ffd(s.iloc[:300], 0.4)
    np.testing.assert_allclose(a.iloc[:300].dropna(), b.dropna())


def test_rolling_beta_recovers():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, 500))
    r = 1.5 * m + rng.normal(0, 0.002, 500)
    assert abs(rolling_beta(r, m).iloc[-1] - 1.5) < 0.1


def test_factor_exposures_recover_and_are_oos():
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    F = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=idx, columns=["Mkt_RF", "SMB"])
    r = pd.Series(0.9 * F.Mkt_RF + 0.4 * F.SMB + rng.normal(0, 0.002, n), index=idx)
    ex = factor_exposures(r, F, window=126)
    last = ex.iloc[-1]
    assert abs(last["beta_Mkt_RF"] - 0.9) < 0.1 and abs(last["beta_SMB"] - 0.4) < 0.1
    assert abs(ex["resid_ret"].dropna()).mean() < 0.004
    # beta at t must not depend on r[t]: perturb final return, final betas unchanged
    r2 = r.copy()
    r2.iloc[-1] += 0.5
    ex2 = factor_exposures(r2, F, window=126)
    assert np.isclose(ex2["beta_Mkt_RF"].iloc[-1], ex["beta_Mkt_RF"].iloc[-1])


def test_cs_rank_is_centered():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2026-01-05", periods=2), list("ABCD")], names=["date", "symbol"])
    p = pd.DataFrame({"x": [1, 2, 3, 4, 4, 3, 2, 1.0]}, index=idx)
    r = cs_rank(p, ["x"])["x_csr"]
    assert np.allclose(r.groupby(level="date").mean(), 0.125)  # pct ranks .25,.5,.75,1 minus .5
    assert r.xs("D", level="symbol").iloc[0] > r.xs("A", level="symbol").iloc[0]


def test_triple_barrier_hits_and_conservative_tie():
    idx = pd.bdate_range("2026-01-05", periods=40)
    n = len(idx)
    base = np.full(n, 100.0)
    df = pd.DataFrame({"open": base, "high": base * 1.001, "low": base * 0.999, "close": base,
                       "adj_close": base, "volume": 1e6}, index=idx)
    # noisy history so sigma > 0, then a spike up on day 30
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df[["open", "close", "adj_close"]] = np.column_stack([c, c, c])
    df["high"], df["low"] = c * 1.002, c * 0.998
    df.iloc[27, df.columns.get_loc("high")] = c[27] * 1.5  # t=25 entry at open[26], bar 27 is k=1
    tb = triple_barrier(df, horizon=5)
    assert tb["tb_label"].iloc[26 - 1] == 1.0
    df.iloc[27, df.columns.get_loc("low")] = c[27] * 0.5  # both barriers in one bar -> stop wins
    tb2 = triple_barrier(df, horizon=5)
    assert tb2["tb_label"].iloc[25] == -1.0
    assert tb2["tb_label"].iloc[-6:].isna().all()


def test_meta_labels():
    side = pd.Series([1, -1, 0, 1])
    tb = pd.Series([1, 1, 1, np.nan])
    m = meta_labels(side, tb)
    assert m.iloc[0] == 1 and m.iloc[1] == 0 and np.isnan(m.iloc[2]) and np.isnan(m.iloc[3])


def test_build_panel_end_to_end(tmp_path, universe):
    from stockpred.data.store import Store

    st = Store(tmp_path / "p.duckdb")
    frames = []
    for i, s in enumerate(universe.symbols + ["SPY"]):
        p = make_prices(n=700, seed=10 + i)
        d = p.reset_index(names="date")
        d["symbol"] = s
        frames.append(d)
    st.upsert_prices(pd.concat(frames))
    panel = build_panel(st, universe, horizon=5, benchmark="SPY", garch=False, regime=False)
    assert {"fwd_ret_5d", "fwd_exc_5d", "fwd_up_5d", "ret_5", "rv_20", "beta_mkt", "ret_5_csr"} <= set(panel.columns)
    assert panel.index.names == ["date", "symbol"]
    last = panel.xs(panel.index.get_level_values("date").max(), level="date")
    assert last["fwd_exc_5d"].isna().all() and last["ret_5"].notna().all()  # features known, label not
    st.close()
