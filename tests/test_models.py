import numpy as np
import pandas as pd
import pytest

from stockpred.models.cv import PurgedKFold, PurgedWalkForward
from stockpred.models.ensemble import oof_predictions, stack_walkforward
from stockpred.models.evaluate import hit_rate, ic_summary, quantile_spread, rank_ic, wilson
from stockpred.models.gbm import GBM
from stockpred.models.labels import cross_sectional_labels
from stockpred.models.linear import LinearModel
from stockpred.models.targets import cs_rank_target
from stockpred.models.train import (TrainConfig, fit_bundle, load_bundle, predict_bundle, run_experiment,
                                    save_bundle, select_features)
from stockpred.models.tracking import log_run
from stockpred.models.uncertainty import DirectionCalibrator, brier, reliability_table, rolling_conformal


def synth_panel(n_dates=1100, n_sym=20, seed=0, signal=0.12, noise_feats=4):
    """Panel with one genuinely predictive feature (`alpha_f`), plus noise features."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(n_sym)]], names=["date", "symbol"])
    n = len(idx)
    alpha_f = rng.normal(size=n)
    noise = {f"noise{k}": rng.normal(size=n) for k in range(noise_feats)}
    y = signal * alpha_f * 0.02 + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"alpha_f": alpha_f, **noise, "fwd_exc_5d": y}, index=idx)
    df["fwd_up_5d"] = (df["fwd_exc_5d"] > 0).astype(float)
    df["fwd_ret_5d"] = df["fwd_exc_5d"]
    # make the last 6 dates unlabeled, like real data
    df.iloc[-6 * n_sym:, df.columns.get_loc("fwd_exc_5d")] = np.nan
    df.iloc[-6 * n_sym:, df.columns.get_loc("fwd_up_5d")] = np.nan
    return df


# ------------------------------------------------------------------ CV
def test_walk_forward_purges_label_overlap():
    p = synth_panel(n_dates=900, n_sym=3)
    cv = PurgedWalkForward(n_splits=3, test_days=100, min_train_days=300, horizon=5, embargo=2)
    ud = np.array(sorted(p.index.get_level_values("date").unique()))
    d = p.index.get_level_values("date")
    folds = list(cv.split(p.index))
    assert len(folds) == 3
    for tr, te in folds:
        tr_max_i = np.searchsorted(ud, d[tr].max())
        te_min_i = np.searchsorted(ud, d[te].min())
        assert te_min_i - tr_max_i >= 5 + 1 + 2  # label window (h+1) + embargo
        assert d[tr].max() < d[te].min()
    # test blocks do not overlap and tile to the end
    all_te = np.concatenate([te for _, te in folds])
    assert len(all_te) == len(set(all_te)) and d[all_te].max() == ud[-1]


def test_walk_forward_too_short_raises():
    p = synth_panel(n_dates=100, n_sym=2)
    with pytest.raises(ValueError):
        list(PurgedWalkForward(2, 50, 200).split(p.index))


def test_purged_kfold_has_gaps_on_both_sides():
    p = synth_panel(n_dates=600, n_sym=2)
    ud = np.array(sorted(p.index.get_level_values("date").unique()))
    d = p.index.get_level_values("date")
    for tr, te in PurgedKFold(4, horizon=5, embargo=3).split(p.index):
        lo_i, hi_i = np.searchsorted(ud, d[te].min()), np.searchsorted(ud, d[te].max())
        tri = np.searchsorted(ud, d[tr])
        before, after = tri[tri < lo_i], tri[tri > hi_i]
        if len(before):
            assert lo_i - before.max() >= 6
        if len(after):
            assert after.min() - hi_i >= 3
        assert not set(tr) & set(te)


# ------------------------------------------------------------------ metrics
def test_rank_ic_perfect_and_inverse():
    p = synth_panel(n_dates=60, n_sym=10)
    y = p["alpha_f"]
    assert rank_ic(y, y).mean() == pytest.approx(1.0)
    assert rank_ic(-y, y).mean() == pytest.approx(-1.0)


def test_ic_summary_and_spread_on_signal():
    p = synth_panel(n_dates=400, n_sym=30)
    y = p["fwd_exc_5d"]
    ic = rank_ic(p["alpha_f"], y)
    s = ic_summary(ic, 5)
    assert s["ic_mean"] > 0.03 and s["ic_t_hac"] > 2
    q = quantile_spread(p["alpha_f"], y)
    assert q["spread"] > 0


def test_wilson_and_hit_rate():
    lo, hi = wilson(60, 100)
    assert lo < 0.6 < hi and 0.49 < lo
    p = synth_panel(n_dates=200, n_sym=10)
    h = hit_rate(p["alpha_f"], p["fwd_exc_5d"])
    assert h["hit_lo"] <= h["hit_rate"] <= h["hit_hi"]


# ------------------------------------------------------------------ models
def test_models_learn_planted_signal_and_ignore_noise():
    p = synth_panel(n_dates=900, n_sym=20, signal=0.5)
    lab = p["fwd_exc_5d"].notna()
    y = cs_rank_target(p["fwd_exc_5d"])
    X = p.drop(columns=["fwd_exc_5d", "fwd_up_5d", "fwd_ret_5d"])
    tr = np.arange(len(p)) < int(len(p) * 0.7)
    te = ~tr & lab.to_numpy()
    for model in (LinearModel("ridge", alpha=10.0), GBM(params={"n_estimators": 200, "min_child_samples": 50})):
        model.fit(X[tr & lab.to_numpy()], y[tr & lab.to_numpy()])
        pred = pd.Series(model.predict(X[te]), index=X.index[te])
        ic = rank_ic(pred, p.loc[te, "fwd_exc_5d"]).mean()
        assert ic > 0.05, type(model).__name__
    g = GBM(params={"n_estimators": 200, "min_child_samples": 50}).fit(X[tr & lab.to_numpy()], y[tr & lab.to_numpy()])
    assert g.importance().index[0] == "alpha_f"
    sh = g.contributions(X.iloc[:5])
    assert sh.shape == (5, X.shape[1] + 1)


def test_shuffled_labels_give_no_edge():
    """Leakage canary: with labels permuted across time within symbol, OOS IC must be ~0."""
    p = synth_panel(n_dates=900, n_sym=20, signal=0.5)
    lab = p["fwd_exc_5d"].notna()
    y = p["fwd_exc_5d"].copy()
    y[lab] = y[lab].sample(frac=1.0, random_state=1).to_numpy()
    X = p.drop(columns=["fwd_exc_5d", "fwd_up_5d", "fwd_ret_5d"])
    yt = cs_rank_target(y)
    cut = int(len(p) * 0.7)
    m = GBM(params={"n_estimators": 150, "min_child_samples": 50}).fit(X.iloc[:cut][lab.iloc[:cut]], yt.iloc[:cut][lab.iloc[:cut]])
    te = lab.iloc[cut:]
    pred = pd.Series(m.predict(X.iloc[cut:][te]), index=X.index[cut:][te])
    assert abs(rank_ic(pred, y.iloc[cut:][te]).mean()) < 0.03


def test_run_experiment_end_to_end_and_bundle(tmp_path):
    p = synth_panel(n_dates=1100, n_sym=20, signal=0.5)
    cfg = TrainConfig(horizon=5, embargo=2, n_splits=3, test_days=100, min_train_days=400, calib_days=60)
    res = run_experiment(p, cfg, tmp_path / "runs")
    m = res.metrics
    assert {"ridge", "lgbm", "stack"} <= set(m.index)
    assert m.loc["lgbm", "ic_mean"] > 0.05 and m.loc["stack", "ic_mean"] > 0.05
    assert (res.stack_weights[["ridge", "lgbm"]].sum(axis=1).round(6) == 1.0).all()
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
    assert "alpha_f" in res.features
    cov = res.extras.get("conformal_coverage")
    assert cov is not None and 0.6 < cov < 0.95  # nominal 0.8

    b = fit_bundle(p, cfg, res)
    d = save_bundle(b, tmp_path / "models")
    b2 = load_bundle(tmp_path / "models")
    last = p.xs(p.index.get_level_values("date").max(), level="date", drop_level=False)
    out = predict_bundle(b2, last)
    assert set(out.columns) >= {"score", "p_up", "exp_excess_ret", "lo", "hi", "label"}
    assert (out["lo"] < out["hi"]).all() and out["p_up"].between(0, 1).all()
    top = out["score"].idxmax()
    assert out.loc[top, "label"] in ("STRONG BUY", "BUY", "NEUTRAL")
    assert (d / "meta.json").exists()


def test_baselines_included_when_text_present():
    p = synth_panel(n_dates=1000, n_sym=10)
    p["mom_12_1"], p["ret_5"] = p["alpha_f"], p["noise0"]
    from stockpred.models.baselines import available_baselines

    b = available_baselines(p)
    assert "momentum_12_1" in b and "reversal_5d" in b and "sentiment_only" not in b


def test_select_features_drops_targets_and_constants():
    p = synth_panel(n_dates=300, n_sym=3)
    p["const"] = 1.0
    p["mostly_nan"] = np.nan
    f = select_features(p)
    assert "const" not in f and "mostly_nan" not in f and "fwd_exc_5d" not in f and "alpha_f" in f


def test_stacking_weights_use_only_past_folds():
    n = 6000
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=300), list("ABCDEFGHIJKLMNOPQRST")],
                                     names=["date", "symbol"])[:n]
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=n), index=idx)
    oof = pd.DataFrame({"good": y + rng.normal(0, 0.5, n), "bad": rng.normal(size=n)}, index=idx)
    fold_id = np.repeat([0, 1, 2], n // 3)
    st, w = stack_walkforward(oof, y, fold_id)
    assert w.loc[0, "good"] == pytest.approx(0.5)  # no history -> equal weight
    assert w.loc[2, "good"] > 0.8 and st.notna().all()


def test_oof_predictions_shapes():
    p = synth_panel(n_dates=700, n_sym=5)
    lab = p["fwd_exc_5d"].notna()
    X = p.drop(columns=["fwd_exc_5d", "fwd_up_5d", "fwd_ret_5d"])[lab]
    y = cs_rank_target(p["fwd_exc_5d"])[lab]
    folds = list(PurgedWalkForward(2, 100, 300, 5, 2).split(X.index))
    oof, fid = oof_predictions({"r": lambda: LinearModel("ridge")}, X, y, folds)
    assert (fid >= 0).sum() == oof["r"].notna().sum()


# ------------------------------------------------------------------ uncertainty
def test_rolling_conformal_coverage_iid():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2018-01-01", periods=500), list("ABCDEFGHIJ")], names=["date", "symbol"])
    pred = pd.Series(rng.normal(size=len(idx)) * 0.01, index=idx)
    y = pred + pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    out = rolling_conformal(pred, y, alpha=0.2, calib_days=100, horizon=5)
    assert 0.76 < out["covered"].mean() < 0.84
    # first usable interval only after horizon+1 realised days (labels must be known)
    first = out["lo"].dropna().index.get_level_values("date").min()
    assert first >= idx.get_level_values("date").unique()[5 + 1]


def test_direction_calibrator_and_reliability():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    up = (rng.uniform(size=5000) < 1 / (1 + np.exp(-2 * s))).astype(float)
    c = DirectionCalibrator().fit(s[:2500], up[:2500])
    p = c.predict(s[2500:])
    assert brier(p, up[2500:]) < brier(np.full(2500, 0.5), up[2500:])
    t = reliability_table(p, up[2500:])
    assert (t["mean_pred"] - t["frac_up"]).abs().max() < 0.12


def test_labels_by_quantile_and_pup_veto():
    s = pd.Series(np.arange(10.0), index=list("abcdefghij"))
    lab = cross_sectional_labels(s)
    assert lab["j"] == "STRONG BUY" and lab["a"] == "STRONG SELL" and lab["e"] == "NEUTRAL"
    veto = cross_sectional_labels(s, pd.Series(0.4, index=s.index))
    assert veto["j"] == "NEUTRAL" and veto["a"] == "STRONG SELL"


def test_run_logger_counts_trials(tmp_path):
    log_run(tmp_path, "a", {"x": 1}, {"m": 1.0}, "fp", n_trials=3)
    path = log_run(tmp_path, "b", {}, {}, "fp", n_trials=2)
    import json

    assert json.loads(path.read_text())["n_trials_total"] == 5


def test_lstm_smoke():
    torch = pytest.importorskip("torch")
    from stockpred.models.sequence import LSTMRegressor, make_lagged

    p = synth_panel(n_dates=260, n_sym=6, signal=0.8)
    lag = make_lagged(p, ["alpha_f", "noise0"], 8)
    lab = p["fwd_exc_5d"].notna()
    y = cs_rank_target(p["fwd_exc_5d"])
    m = LSTMRegressor(["alpha_f", "noise0"], seq_len=8, hidden=8, epochs=3, batch=256)
    m.fit(lag[lab], y[lab])
    pred = m.predict(lag)
    assert pred.shape == (len(p),) and np.isfinite(pred).all()
    # first rows have NaN lags -> must not produce NaN predictions
    assert torch is not None
