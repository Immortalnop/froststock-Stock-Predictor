import numpy as np
import pandas as pd
import pytest

from stockpred.portfolio.construct import ConstructConfig, build_target
from stockpred.portfolio.optimize import (black_litterman, evidence_scale, hrp, inverse_vol, ledoit_wolf_cov,
                                          mean_variance, min_variance, signal_to_alpha)
from stockpred.portfolio.paper import PaperAccount, alpaca_orders_payload
from stockpred.portfolio.risk import (RiskLimits, check_limits, hist_var_cvar, monte_carlo_var_cvar, scale_to_limits,
                                      stress_replay, stress_shock)
from stockpred.portfolio.sizing import drawdown_throttle, kelly_fraction, vol_target_scale
from stockpred.data.store import Store
from conftest import make_prices


def _rets(n=500, k=12, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (n, 1))
    vols = np.linspace(0.008, 0.025, k)
    x = f * np.linspace(0.6, 1.4, k) + rng.normal(0, 1, (n, k)) * vols
    return pd.DataFrame(x, index=pd.bdate_range("2021-01-01", periods=n), columns=[f"A{i}" for i in range(k)])


def test_cov_shrinkage_psd_and_inverse_vol():
    cov = ledoit_wolf_cov(_rets())
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0
    w = inverse_vol(cov)
    assert w.sum() == pytest.approx(1) and w.iloc[0] > w.iloc[-1]  # low-vol name gets more weight


def test_min_variance_beats_equal_weight():
    cov = ledoit_wolf_cov(_rets())
    w = min_variance(cov)
    ew = pd.Series(1 / len(cov), index=cov.index)
    assert w.sum() == pytest.approx(1) and (w >= -1e-9).all()
    assert w @ cov @ w <= ew @ cov @ ew + 1e-12


def test_hrp_weights_valid_and_risk_aware():
    cov = ledoit_wolf_cov(_rets())
    w = hrp(cov)
    assert w.sum() == pytest.approx(1) and (w > 0).all()
    assert w.iloc[0] > w.iloc[-1]


def test_mean_variance_respects_constraints():
    r = _rets()
    cov = ledoit_wolf_cov(r)
    mu = pd.Series(np.linspace(-0.001, 0.001, len(cov)), index=cov.index)
    betas = pd.Series(np.linspace(0.6, 1.4, len(cov)), index=cov.index)
    w = mean_variance(mu, cov, gross=1.0, net_max=0.02, max_w=0.15, betas=betas, beta_max=0.05)
    assert w.abs().sum() <= 1.0 + 1e-6 and abs(w.sum()) <= 0.02 + 1e-6 and w.abs().max() <= 0.15 + 1e-6
    assert abs(w @ betas) <= 0.05 + 1e-6
    assert np.corrcoef(w, mu)[0, 1] > 0.8  # follows the alpha ranking
    # turnover penalty shrinks trades toward the previous book
    w2 = mean_variance(mu, cov, w_prev=w * 0.5, turnover_penalty=1.0, net_max=0.02, max_w=0.15)
    assert (w2 - w * 0.5).abs().sum() < (w - w * 0.5).abs().sum()


def test_black_litterman_tilts_toward_views():
    cov = ledoit_wolf_cov(_rets())
    mkt = pd.Series(1 / len(cov), index=cov.index)
    base = black_litterman(cov, mkt, pd.Series(0.0, index=cov.index))
    view = pd.Series(0.0, index=cov.index)
    view.iloc[0] = 0.02
    post = black_litterman(cov, mkt, view)
    assert post.iloc[0] > base.iloc[0]
    assert post.iloc[0] - base.iloc[0] < 0.02 / 1  # shrunk, not adopted wholesale
    hi = black_litterman(cov, mkt, view, view_conf=0.9).iloc[0]
    assert hi > post.iloc[0]


def test_alpha_mapping_and_evidence_gate():
    s = pd.Series([1.0, 0.0, -1.0, 2.0, -2.0])
    a = signal_to_alpha(s, pd.Series(0.01, index=s.index), ic=0.05, horizon=4)
    assert a.corr(s) == pytest.approx(1) and a.abs().max() < 0.05
    assert evidence_scale(-0.2) == 0 and evidence_scale(None) == 0 and evidence_scale(1.0) == 0.5
    assert evidence_scale(5.0) == 1.0


def test_no_evidence_means_no_positions():
    r = _rets()
    score = pd.Series(np.random.default_rng(0).normal(size=r.shape[1]), index=r.columns)
    flat = build_target(score, r, ic_t=-0.3, cfg=ConstructConfig())
    assert flat.weights.abs().sum() == 0 and flat.gate == 0
    live = build_target(score, r, ic_t=None, cfg=ConstructConfig(ignore_gate=True))
    assert live.weights.abs().sum() > 0.2 and live.report.ok


def test_var_cvar_known_answers():
    r = pd.Series(np.concatenate([np.full(90, 0.01), np.full(10, -0.05)]))
    h = hist_var_cvar(r, 0.95)
    assert h["var"] == pytest.approx(0.05) and h["cvar"] == pytest.approx(0.05)
    rets = _rets()
    w = pd.Series(1 / rets.shape[1], index=rets.columns)
    mc = monte_carlo_var_cvar(w, rets, 0.95, horizon=1, n_sims=20000)
    hist = hist_var_cvar((rets * w).sum(axis=1), 0.95)
    assert mc["cvar"] > mc["var"] > 0 and mc["var"] == pytest.approx(hist["var"], rel=0.35)
    mc5 = monte_carlo_var_cvar(w, rets, 0.95, horizon=5)
    assert mc5["var"] > mc["var"]  # longer horizon, bigger loss


def test_stress_and_limits():
    idx = pd.bdate_range("2020-02-10", "2020-04-01")
    rets = pd.DataFrame({"A": -0.01, "B": 0.005}, index=idx)
    w = pd.Series({"A": 0.5, "B": -0.5})
    st = stress_replay(w, rets)
    assert st.loc["covid_crash", "pnl"] < 0  # long the loser, short the winner
    assert stress_shock(w, pd.Series({"A": 1.2, "B": 0.8}), -0.10) == pytest.approx(-0.10 * (0.6 - 0.4))
    lim = RiskLimits(max_gross=1.0, max_net=0.05, max_name=0.1)
    big = pd.Series({"A": 0.4, "B": -0.1, "C": 0.3})
    rep = check_limits(big, lim)
    assert not rep.ok and any("gross" in v or "net" in v or "name" in v for v in rep.violations)
    fixed = scale_to_limits(big, lim)
    assert fixed.abs().max() <= 0.1 + 1e-12 and fixed.abs().sum() <= 1.0


def test_sizing_overlays():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 400))
    sc = vol_target_scale(r, target_vol=0.10)
    assert (sc.dropna() <= 3.0).all()
    scaled = (r * sc.shift(1)).dropna()
    assert abs(scaled.std() * np.sqrt(252) - 0.10) < 0.04
    assert kelly_fraction(-0.01, 0.01) == 0 and kelly_fraction(0.001, 0.0004) == pytest.approx(0.625)
    eq = pd.Series([1.0, 1.1, 1.0, 1.0, 1.0])
    t = drawdown_throttle(eq, 0.05, 0.15)
    assert t.iloc[0] == 1 and 0 < t.iloc[-1] < 1


def _paper_store():
    st = Store()
    frames = []
    for i, s in enumerate(["AAA", "BBB", "SPY"]):
        d = make_prices(n=60, seed=i, start="2024-01-02").reset_index(names="date")
        d["symbol"] = s
        frames.append(d)
    st.upsert_prices(pd.concat(frames))
    return st, pd.concat(frames)


def test_paper_account_round_trip_conserves_value():
    st, px = _paper_store()
    acct = PaperAccount(st, initial_cash=100_000.0)
    d0 = px.date.sort_values().unique()[30]
    q = acct.rebalance(pd.Series({"AAA": 0.3, "BBB": -0.2}), d0)
    assert len(q) == 2 and acct.positions().empty  # nothing filled before the next bar
    assert acct.settle() == 2
    pos = acct.positions()
    assert pos["AAA"] > 0 > pos["BBB"]
    fills = acct.orders("filled")
    d1 = pd.Timestamp(d0) + pd.offsets.BDay(1)
    assert (pd.to_datetime(fills.fill_date) == d1).all()  # filled at the NEXT bar, never the decision bar
    assert (fills.cost > 0).all()
    row = acct.mark(d1)
    assert row["nav"] < 100_000 * 1.05 and row["nav"] > 100_000 * 0.9
    assert abs(row["net"] - 0.1) < 0.05
    # flatten
    acct.rebalance(pd.Series(dtype=float), d1)
    acct.settle()
    assert acct.positions().empty
    st.close()


def test_paper_rebalance_cancels_stale_intent_and_settle_is_idempotent():
    st, px = _paper_store()
    acct = PaperAccount(st, initial_cash=50_000.0)
    d0 = px.date.sort_values().unique()[20]
    acct.rebalance(pd.Series({"AAA": 0.5}), d0)
    acct.rebalance(pd.Series({"BBB": 0.5}), d0)
    assert set(acct.orders("pending").symbol) == {"BBB"}
    assert acct.settle() == 1 and acct.settle() == 0
    st.close()


def test_alpaca_payload_is_market_on_open():
    o = alpaca_orders_payload(pd.Series({"AAA": 0.1, "BBB": -0.1}), 100_000, pd.Series({"AAA": 100.0, "BBB": 50.0}),
                              pd.Series({"BBB": -50.0}))
    d = {x["symbol"]: x for x in o}
    assert d["AAA"]["side"] == "buy" and d["AAA"]["qty"] == "100" and d["AAA"]["time_in_force"] == "opg"
    assert d["BBB"]["side"] == "sell" and d["BBB"]["qty"] == "150"
