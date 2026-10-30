# stockpred

News + price signal research platform. It ingests prices and news, extracts structured sentiment with an LLM, builds point-in-time features, and evaluates models with purged walk-forward validation. The point is to **measure whether a signal exists**, not to assume one.

> Research tool. Output is a statistical estimate, not financial advice. On the bundled 30-stock large-cap universe the current price-only models show **no statistically significant out-of-sample edge** (see "Results so far"). That is an expected, honest result for efficient large caps.

## Layout

```
stockpred/
  config/     typed settings (env STOCKPRED_*), universe + aliases
  data/       RSS/GDELT news, yfinance prices, FRED + Fama-French, DuckDB store, MinHash dedup,
              entity linking, point-in-time S&P 500 membership
  nlp/        LLM providers (Ollama/OpenAI/Anthropic), validated extraction, lexicon + FinBERT
              baselines, TF-IDF novelty, confidence calibration
  features/   technical, volatility (+ leak-free rolling GARCH/GJR), fractional differencing,
              cross-sectional ranks, rolling beta, Fama-French residuals, Markov-switching regime,
              decayed text features
  models/     purged CV, targets (excess return, triple-barrier, meta-labels), ridge, LightGBM (+SHAP),
              LSTM (torch), walk-forward stacking, Optuna tuning, conformal intervals, calibration
  backtest/   next-open, cost-aware long/short engine (spread, sqrt impact, borrow), PSR/deflated Sharpe,
              PBO (CSCV), White Reality Check, stationary-bootstrap CIs, factor attribution, decay + capacity
  portfolio/  Ledoit-Wolf, inverse-vol, min-var, HRP, Black-Litterman, constrained mean-variance, vol-target /
              Kelly / drawdown sizing, VaR/CVaR (historical, parametric, Monte Carlo), stress tests, limits,
              paper trading (local simulated broker + Alpaca paper payloads)
  mlops/      PSI/KS feature drift, retrain policy
  serve/      live pipeline, Rich dashboard, prediction ledger, scheduler, FastAPI
  cli.py      `python -m stockpred <command>`
tests/        100+ tests incl. causality (truncation), shuffled-label leakage canary, golden backtest
docs/         MODEL_CARD.md
_legacy_v0/   original prototype, kept for reference
```

## Quick start

```bash
pip install -r requirements.txt            # torch / transformers optional
python -m stockpred ingest-prices --years 10
python -m stockpred ingest-macro                       # VIX, yields, Fama-French
python -m stockpred build-panel --text none            # features + targets -> artifacts/panel.parquet
python -m stockpred train                              # walk-forward report + saves artifacts/models/latest
python -m stockpred ingest-news                        # RSS into the store
ollama pull llama3.2 && python -m stockpred extract    # LLM extraction of pending articles
python -m stockpred build-panel --text llm && python -m stockpred train
python -m stockpred live                               # dashboard (add --once, --no-ui, --force)
python -m stockpred backtest --curves                  # net-of-cost backtest of every OOS signal + overfitting tests
python -m stockpred paper cycle [--ignore-gate]        # settle fills, mark NAV, queue the next rebalance
python -m stockpred risk                               # VaR/CVaR/stress for the paper book
python -m stockpred monitor                            # drift, live IC from the ledger, retrain recommendation
python -m stockpred serve --paper                      # scheduler + FastAPI on 127.0.0.1:8000 (docker compose up)
python -m stockpred doctor                             # LLM / data status
python -m pytest
```

Other providers: `STOCKPRED_LLM_PROVIDER=anthropic STOCKPRED_ANTHROPIC_API_KEY=... STOCKPRED_LLM_MODEL=claude-haiku-5-5`.
`train` options: `--lstm`, `--tune N` (Optuna on the first fold's training data only), `--splits`, `--test-days`.

## Methodology highlights

- **Point-in-time news.** Each article has `published_at`, `ingested_at` and `available_at`. It affects only the first trading day whose 16:00 ET close is at or after `available_at`. Undated items are dropped, never defaulted to "now". Backfilled history uses `published_at + latency`.
- **Tradeable labels.** Target at date t = adjusted open(t+1+h) / open(t+1) - 1, minus the benchmark. No same-bar fills. Models train on a winsorised cross-sectional rank of it; metrics use the raw excess return.
- **Purged validation.** Walk-forward with a gap of `horizon + 1 + embargo` days; PurgedKFold also available. Stacking weights for fold k use only folds < k.
- **Causality tests.** Every feature builder is tested by truncating the series and checking the last row is unchanged. GARCH/regime parameters are refit periodically, then held fixed (no refit leakage).
- **Baselines must be beaten.** 12-1 momentum, 5-day reversal, sentiment-only, and the v0 hand-weighted score (faithfully reproduced, including its relative-buzz flaw).
- **Uncertainty.** Rolling split-conformal intervals use only residuals whose labels were already realised; direction probabilities are isotonic-calibrated and checked with Brier score and reliability tables.
- **LLM hygiene.** JSON schema validation with clipping, missing confidence = 0 (not 0.5), failures never cached, cache key = hash(model, prompt version, full text, universe), tickers rejected unless the text evidences them, optional self-consistency (`STOCKPRED_LLM_SAMPLES`), and `mask_entities` for look-ahead contamination ablations.
- **Honest backtesting.** Orders fill at the next open and pay commission + half-spread + sqrt-law impact (vs ADV)
  + borrow. Reported with bootstrap Sharpe CIs, deflated Sharpe over every logged trial, PBO via CSCV, White's
  Reality Check, and factor-regression alpha (HAC t). `--curves` adds signal-decay and capacity (AUM) curves.
- **Evidence-gated portfolio.** Alpha = IC x vol x z (Grinold), scaled by walk-forward evidence: with IC t <= 0 the
  book is flat. `--ignore-gate` trades the signal anyway so the paper ledger measures the null.
- **Live ground truth.** Every dashboard cycle logs predictions; `serve.ledger` joins them to realised excess
  returns (same fill/exit convention as the backtest) and reports live rank IC with HAC t, which also feeds the
  retrain policy.
- **Fixed v0 bugs.** Broken imports, `mktime` vs `timegm`, articles marked seen then dropped, signals overwritten each cycle, 6-day momentum window, blocking yfinance in the event loop, wrong cache key, dead Reuters feed, stub OpenAI path, alt-screen `--once`, asymmetric score bar.

## Results so far (30 US large caps, 2016-2026, price/macro features, h=5d)

Walk-forward signal quality, then the same OOS signals run through the cost-aware backtester (quintile long/short,
5-day rebalance, USD 1M, default costs):

| signal | rank IC | HAC t | gross Sharpe | net Sharpe [95% CI] | DSR |
|---|---|---|---|---|---|
| 12-1 momentum baseline | 0.025 | 1.3 | 0.34 | 0.30 [-0.33, 0.92] | 0.23 |
| LightGBM | 0.000 | 0.0 | -0.32 | -0.42 [-1.50, 0.61] | 0.01 |
| Ridge | -0.006 | -0.4 | -0.14 | -0.25 [-1.11, 0.79] | 0.02 |
| stack | -0.003 | -0.2 | -0.17 | -0.26 [-1.24, 0.68] | 0.02 |

PBO 0.50, Reality Check p = 0.24, 12 logged trials. Direction Brier 0.2502 vs 0.2500 climatology; conformal
coverage 78.8% at a nominal 80%. The machinery works and says the same thing every time: **there is no edge to
exploit from these features on this universe.** The dashboard, API and paper book all carry that warning. News
features need accumulated extractions before they can be judged (RSS history is shallow; GDELT covers ~3 months).

## Known limitations

- Yahoo adjusted prices are a restated snapshot; use `ingest-prices --full` periodically. Universe is a static list (survivorship-biased); `data/universe_sp500.py` reconstructs historical membership but is not yet wired into `build-panel`.
- Ken French factors publish weeks late, so they are research-only (`build-panel --factors`).
- Paper fills use next-open prices with the model cost; splits are not handled in the local paper account. The Alpaca client builds paper-endpoint orders but is untested against the live service.
- The API has no authentication (bound to localhost in `docker-compose.yml`). No Streamlit/Plotly UI: the Rich dashboard and JSON API are the interfaces.
- Overlapping 5-day labels inflate naive hit-rate CIs; the HAC t-stat on IC is the number to trust.
- LSTM is evaluated in `train --lstm` but not persisted in the final bundle.
