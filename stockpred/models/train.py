"""Experiment orchestration: purged walk-forward evaluation of baselines vs learned models,
stacked ensemble, calibrated direction probabilities, conformal intervals, final fit + bundle."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from stockpred.features.build import TARGET_PREFIX
from stockpred.models.baselines import available_baselines, buy_and_hold_excess
from stockpred.models.cv import PurgedWalkForward
from stockpred.models.ensemble import oof_predictions, stack_walkforward
from stockpred.models.evaluate import evaluate_signal
from stockpred.models.gbm import GBM
from stockpred.models.labels import cross_sectional_labels
from stockpred.models.linear import LinearModel
from stockpred.models.targets import cs_rank_target, winsorize_cs
from stockpred.models.tracking import fingerprint, log_run
from stockpred.models.uncertainty import DirectionCalibrator, brier, rolling_conformal

log = logging.getLogger(__name__)
LSTM_FEATURES = ["ret_1", "ret_5", "ret_21", "rsi_14", "macd_hist", "atr_pct", "volume_z_20",
                 "rv_20", "gap", "intraday_ret", "sent_ew1d", "sent_ew5d", "buzz_z"]


@dataclass
class TrainConfig:
    horizon: int = 5
    embargo: int = 2
    n_splits: int = 6
    test_days: int = 126
    min_train_days: int = 504
    use_lstm: bool = False
    tune_trials: int = 0
    conformal_alpha: float = 0.2
    calib_days: int = 126
    seed: int = 7
    max_nan_frac: float = 0.98
    ridge_alpha: float = 300.0

    @property
    def target(self) -> str:
        return f"fwd_exc_{self.horizon}d"

    @property
    def up_target(self) -> str:
        return f"fwd_up_{self.horizon}d"


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    oof: pd.DataFrame
    stack: pd.Series
    stack_weights: pd.DataFrame
    features: list[str]
    config: TrainConfig
    conformal: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


class Subset:
    """Restrict a model to a column subset so models with different inputs share one frame."""

    def __init__(self, model, cols: list[str]):
        self.model, self.cols = model, cols

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.cols])


def select_features(panel: pd.DataFrame, max_nan_frac: float = 0.98) -> list[str]:
    cols = []
    for c in panel.columns:
        if c.startswith(TARGET_PREFIX) or not pd.api.types.is_numeric_dtype(panel[c]):
            continue
        if panel[c].isna().mean() > max_nan_frac or panel[c].nunique(dropna=True) < 2:
            continue
        cols.append(c)
    return cols


def make_factories(cfg: TrainConfig, feats: list[str], gbm_params: dict | None, lstm_cols: list[str] | None):
    f = {
        "ridge": lambda: Subset(LinearModel("ridge", alpha=cfg.ridge_alpha), feats),
        "lgbm": lambda: Subset(GBM(params={**(gbm_params or {}), "random_state": cfg.seed}), feats),
    }
    if lstm_cols:
        from stockpred.models.sequence import LSTMRegressor

        base = [c.rsplit("__l", 1)[0] for c in lstm_cols if c.endswith("__l0")]
        f["lstm"] = lambda: Subset(LSTMRegressor(base, seq_len=20, seed=cfg.seed), lstm_cols_all(base))
    return f


def lstm_cols_all(base: list[str], seq_len: int = 20) -> list[str]:
    return [f"{c}__l{k}" for k in range(seq_len) for c in base]


def prepare(panel: pd.DataFrame, cfg: TrainConfig):
    """Training target = winsorised, cross-sectionally ranked benchmark-relative return."""
    y_raw = panel[cfg.target]
    y_train = cs_rank_target(winsorize_cs(y_raw))
    return y_raw, y_train


def run_experiment(panel: pd.DataFrame, cfg: TrainConfig, runs_dir: Path | None = None) -> ExperimentResult:
    panel = panel.sort_index()
    feats = select_features(panel, cfg.max_nan_frac)
    y_raw, y_train = prepare(panel, cfg)
    X = panel[feats].copy()

    lstm_cols = None
    if cfg.use_lstm:
        try:
            from stockpred.models.sequence import make_lagged

            base = [c for c in LSTM_FEATURES if c in panel.columns]
            lag = make_lagged(panel, base, 20)
            X = pd.concat([X, lag], axis=1)
            lstm_cols = [f"{c}__l0" for c in base]
        except Exception as exc:  # torch missing etc.
            log.warning("LSTM disabled: %s", exc)

    labeled = y_train.notna()
    Xl, ytl = X[labeled], y_train[labeled]
    cv = PurgedWalkForward(cfg.n_splits, cfg.test_days, cfg.min_train_days, cfg.horizon, cfg.embargo)
    folds = list(cv.split(Xl.index))

    gbm_params, n_trials = None, 1
    if cfg.tune_trials > 0:
        from stockpred.models.tune import tune_gbm

        tr0 = folds[0][0]  # tune on the first fold's training data only: never on any test fold
        gbm_params, best_ic, n_trials = tune_gbm(
            Xl.iloc[tr0][feats], ytl.iloc[tr0], y_raw[labeled].iloc[tr0], cfg.horizon, cfg.embargo,
            cfg.tune_trials, seed=cfg.seed)
        log.info("tuned lgbm: inner IC %.4f params %s", best_ic, gbm_params)

    fac = make_factories(cfg, feats, gbm_params, lstm_cols)
    oof, fold_id = oof_predictions(fac, Xl, ytl, folds)
    stack, weights = stack_walkforward(oof, ytl, fold_id)

    covered = fold_id >= 0
    yr = y_raw[labeled]
    rows = {}
    for name in oof.columns:
        rows[name] = evaluate_signal(oof.loc[covered, name], yr[covered], cfg.horizon)
    rows["stack"] = evaluate_signal(stack[covered], yr[covered], cfg.horizon)
    base_scores = available_baselines(panel)
    for name, s in base_scores.items():
        s = s.reindex(Xl.index)
        rows[f"baseline:{name}"] = evaluate_signal(s[covered], yr[covered], cfg.horizon)
    metrics = pd.DataFrame(rows).T
    metrics["n_features"] = len(feats)

    # calibrated direction probability + conformal intervals on the (honest) stacked OOF
    extras: dict = {"buy_hold_mean_excess": buy_and_hold_excess(panel, cfg.target),
                    "n_obs_oos": int(covered.sum()), "folds": len(folds)}
    conformal = None
    try:
        s_ok = stack[covered].dropna()
        up = panel[cfg.up_target].reindex(s_ok.index)
        half = len(s_ok) // 2
        cal = DirectionCalibrator().fit(s_ok.iloc[:half], up.iloc[:half])
        p_hold = cal.predict(s_ok.iloc[half:])
        extras["brier_direction"] = brier(p_hold, up.iloc[half:])
        extras["brier_climatology"] = brier(np.full(len(p_hold), up.iloc[:half].mean()), up.iloc[half:])
        rmap = LinearRegression().fit(s_ok.to_frame(), yr.reindex(s_ok.index).fillna(0.0))
        exp_ret = pd.Series(rmap.predict(s_ok.to_frame()), index=s_ok.index)
        conformal = rolling_conformal(exp_ret, yr.reindex(s_ok.index), cfg.conformal_alpha,
                                      cfg.calib_days, cfg.horizon)
        extras["conformal_coverage"] = float(conformal["covered"].mean())
        extras["conformal_target"] = 1 - cfg.conformal_alpha
    except Exception as exc:
        log.warning("uncertainty step skipped: %s", exc)

    res = ExperimentResult(metrics, oof, stack, weights, feats, cfg, conformal, extras)
    if runs_dir is not None:
        log_run(runs_dir, "experiment", asdict(cfg),
                {"metrics": metrics[["ic_mean", "ic_t_hac", "hit_rate", "q_spread"]].to_dict("index"),
                 **{k: v for k, v in extras.items() if isinstance(v, (int, float))}},
                fingerprint(panel), n_trials=max(1, n_trials) * len(fac))
    return res


# ---------------------------------------------------------------------------- final bundle
@dataclass
class Bundle:
    models: dict
    weights: dict[str, float]
    features: list[str]
    config: TrainConfig
    trained_through: pd.Timestamp
    calibrator: DirectionCalibrator
    ret_map: LinearRegression
    conformal_q: float
    metrics: dict


def fit_bundle(panel: pd.DataFrame, cfg: TrainConfig, result: ExperimentResult,
               gbm_params: dict | None = None) -> Bundle:
    """Fit final models on all labelled rows. Blend weights = positive-LS on all honest OOF."""
    panel = panel.sort_index()
    feats = result.features
    y_raw, y_train = prepare(panel, cfg)
    lab = y_train.notna()
    X = panel.loc[lab, feats]
    fac = {k: v for k, v in make_factories(cfg, feats, gbm_params, None).items()}
    models = {k: make().fit(X, y_train[lab]) for k, make in fac.items()}
    cols = [c for c in result.oof.columns if c in models]
    oof = result.oof[cols]
    m = oof.notna().all(axis=1) & y_train.reindex(oof.index).notna()
    if m.sum() > 500:
        w = LinearRegression(positive=True, fit_intercept=False).fit(oof[m], y_train.reindex(oof.index)[m]).coef_
        w = w / w.sum() if w.sum() > 1e-12 else np.full(len(cols), 1 / len(cols))
    else:
        w = np.full(len(cols), 1 / len(cols))
    weights = dict(zip(cols, map(float, w)))
    st = (oof[m] * pd.Series(weights)).sum(axis=1)
    up = panel[cfg.up_target].reindex(st.index)
    cal = DirectionCalibrator().fit(st, up)
    ret_map = LinearRegression().fit(st.to_frame(), y_raw.reindex(st.index).fillna(0.0))
    resid = (y_raw.reindex(st.index) - ret_map.predict(st.to_frame())).abs().dropna()
    q = float(np.quantile(resid.iloc[-cfg.calib_days * 20 :], 1 - cfg.conformal_alpha))
    return Bundle(models, weights, feats, cfg, panel.index.get_level_values("date").max(), cal, ret_map, q,
                  result.metrics[["ic_mean", "ic_t_hac", "hit_rate"]].to_dict("index"))


def save_bundle(b: Bundle, directory: Path, name: str = "latest") -> Path:
    directory = Path(directory) / name
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, directory / "bundle.joblib")
    (directory / "meta.json").write_text(json.dumps({
        "trained_through": str(b.trained_through.date()), "weights": b.weights,
        "n_features": len(b.features), "target": b.config.target, "horizon": b.config.horizon,
        "conformal_halfwidth": b.conformal_q, "metrics": b.metrics}, indent=2, default=float))
    return directory


def load_bundle(directory: Path, name: str = "latest") -> Bundle:
    return joblib.load(Path(directory) / name / "bundle.joblib")


def predict_bundle(b: Bundle, X: pd.DataFrame) -> pd.DataFrame:
    """X: panel rows (index date,symbol) containing b.features. Returns score, p_up, expected excess
    return with conformal interval, and quantile-based labels (per date)."""
    X = X[b.features]
    preds = pd.DataFrame({k: m.predict(X) for k, m in b.models.items()}, index=X.index)
    score = (preds * pd.Series(b.weights)).sum(axis=1)
    out = pd.DataFrame({"score": score}, index=X.index)
    out["p_up"] = b.calibrator.predict(score.to_numpy())
    out["exp_excess_ret"] = b.ret_map.predict(score.to_frame())
    out["lo"] = out["exp_excess_ret"] - b.conformal_q
    out["hi"] = out["exp_excess_ret"] + b.conformal_q
    labels = [cross_sectional_labels(g["score"], g["p_up"]) for _, g in out.groupby(level="date")]
    out["label"] = pd.concat(labels).reindex(out.index)
    return out
