import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from stockpred.backtest.costs import ZERO_COST, CostModel
from stockpred.backtest.engine import BacktestConfig, run_backtest, target_weights
from stockpred.backtest.metrics import (deflated_sharpe, expected_max_sharpe, factor_attribution, pbo_cscv,
                                        perf_stats, reality_check)
from stockpred.models.cv import CombinatorialPurgedCV

SYMS = [f"S{i:02d}" for i in range(20)]


def _prices(n=400, seed=3):
    frames = []
    for i, s in enumerate(SYMS):
        d = make_prices(n=n, seed=seed + i, start="2022-01-03", drift=0.0, vol=0.012).reset_index(names="date")
        d["symbol"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _oracle_scores(prices, noise=0.0, seed=0):
    """Score = true next-day open-to-open return (+ noise). A perfect-foresight signal."""
    op = prices.pivot(index="date", columns="symbol", values="open")
    fwd = op.shift(-2) / op.shift(-1) - 1  # score at close t earns open(t+1)->open(t+2)
    rng = np.random.default_rng(seed)
    sc = fwd + rng.normal(0, noise, fwd.shape)
    return sc.stack().rename("score").rename_axis(["date", "symbol"])


def test_target_weights_dollar_neutral_and_gross():
    s = pd.Series(np.arange(20, dtype=float), index=SYMS)
    w = target_weights(s, BacktestConfig(quantile=0.2, gross=1.0))
    assert w.sum() == pytest.approx(0.0) and w.abs().sum() == pytest.approx(1.0)
    assert (w[w > 0].index == SYMS[-4:]).all() and (w[w < 0].index == SYMS[:4]).all()
    assert target_weights(s.iloc[:5], BacktestConfig()).abs().sum() == 0  # too few names -> flat


def test_perfect_foresight_is_hugely_profitable_and_random_is_not():
    px = _prices()
    cfg = BacktestConfig(rebalance_days=1, cost=ZERO_COST)
    good = perf_stats(run_backtest(_oracle_scores(px), px, cfg).net_ret)
    assert good["sharpe"] > 10
    rnd = np.random.default_rng(1).normal(size=len(_oracle_scores(px)))
    bad = perf_stats(run_backtest(pd.Series(rnd, _oracle_scores(px).index), px, cfg).net_ret)
    assert abs(bad["sharpe"]) < 2.5


def test_no_lookahead_signal_uses_next_open():
    """A score equal to the *same-bar* open->open return is not tradeable: shifting it by one bar must
    destroy the oracle edge, while the correctly aligned oracle keeps it."""
    px = _prices()
    op = px.pivot(index="date", columns="symbol", values="open")
    same = (op.shift(-1) / op - 1).stack().rename("score").rename_axis(["date", "symbol"])  # earned before fill
    cfg = BacktestConfig(rebalance_days=1, cost=ZERO_COST)
    assert abs(perf_stats(run_backtest(same, px, cfg).net_ret)["sharpe"]) < 3


def test_costs_reduce_returns_and_scale_with_turnover():
    px = _prices()
    sc = _oracle_scores(px, noise=0.01)
    free = run_backtest(sc, px, BacktestConfig(rebalance_days=1, cost=ZERO_COST))
    paid = run_backtest(sc, px, BacktestConfig(rebalance_days=1, cost=CostModel(5, 10, 0.1, 50)))
    assert paid.net_ret.sum() < free.net_ret.sum()
    assert (paid.cost > 0).any() and paid.borrow.min() >= 0
    slow = run_backtest(sc, px, BacktestConfig(rebalance_days=10, cost=CostModel(5, 10, 0.1, 50)))
    assert slow.avg_annual_turnover < paid.avg_annual_turnover
    smooth = run_backtest(sc, px, BacktestConfig(rebalance_days=1, smooth=0.2, cost=CostModel(5, 10, 0.1, 50)))
    assert smooth.avg_annual_turnover < paid.avg_annual_turnover


def test_impact_grows_with_aum():
    px = _prices()
    sc = _oracle_scores(px, noise=0.01)
    small = run_backtest(sc, px, BacktestConfig(rebalance_days=1, aum=1e5))
    big = run_backtest(sc, px, BacktestConfig(rebalance_days=1, aum=1e10))
    assert big.cost.sum() > small.cost.sum()


def test_golden_backtest_regression():
    """Fixed seed + data: guards against accidental changes to timing, weighting or cost accounting."""
    px = _prices()
    r = run_backtest(_oracle_scores(px, noise=0.02, seed=5), px, BacktestConfig(rebalance_days=5))
    assert len(r.net_ret) > 300
    assert r.net_ret.sum() == pytest.approx(GOLDEN["net_sum"], rel=1e-6)
    assert r.avg_annual_turnover == pytest.approx(GOLDEN["turnover"], rel=1e-6)


GOLDEN = {"net_sum": 0.7364302146195442, "turnover": 40.446582823228475}  # recorded from a reviewed run


def test_psr_dsr_basics():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    d1, d100 = deflated_sharpe(r, 1), deflated_sharpe(r, 100)
    assert d100["dsr"] < d1["dsr"]  # more trials -> more deflation
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert d1["psr0"] == pytest.approx(d1["dsr"])  # one trial = no deflation


def test_pbo_detects_overfitting_in_pure_noise():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1000, 40)))
    assert pbo_cscv(noise)["pbo"] > 0.3  # picking the IS winner of pure noise is a coin flip
    skill = noise.copy()
    skill[0] = skill[0] + 0.003  # a genuinely better strategy survives out of sample
    assert pbo_cscv(skill)["pbo"] < 0.2


def test_reality_check_pvalues():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 0.01, (800, 30)))
    assert reality_check(noise, n_boot=200)["rc_pvalue"] > 0.05
    noise[3] += 0.004
    assert reality_check(noise, n_boot=200)["rc_pvalue"] < 0.05


def test_factor_attribution_recovers_alpha_and_beta():
    rng = np.random.default_rng(3)
    mkt = pd.Series(rng.normal(0.0004, 0.01, 1500), index=pd.bdate_range("2018-01-01", periods=1500))
    strat = 0.0005 + 0.8 * mkt + rng.normal(0, 0.002, 1500)
    out = factor_attribution(strat, pd.DataFrame({"MKT": mkt}))
    assert out["beta_MKT"] == pytest.approx(0.8, abs=0.05)
    assert out["alpha_ann"] == pytest.approx(0.0005 * 252, rel=0.2) and out["alpha_t"] > 3


def test_cpcv_counts_and_purging():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=300), ["A", "B"]],
                                     names=["date", "symbol"])
    cv = CombinatorialPurgedCV(6, 2, horizon=5, embargo=2)
    splits = list(cv.split(idx))
    assert len(splits) == cv.n_splits == 15 and cv.n_paths == 5
    d = idx.get_level_values("date")
    for tr, te in splits:
        assert not set(tr) & set(te)
        t0 = d[te].min()
        # no training row's label window ([t+1, t+1+h]) may overlap a test block start
        before = d[tr][d[tr] < t0]
        if len(before):
            assert before.max() <= pd.bdate_range(end=t0, periods=8)[0]
