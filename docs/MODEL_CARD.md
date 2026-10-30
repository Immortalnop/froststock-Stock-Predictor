# Model card: stockpred stack

**Status: research instrument. No demonstrated out-of-sample edge. Not financial advice.**

## What it is
A cross-sectional ranking model over a 30-stock US large-cap universe. It predicts the 5-trading-day
open-to-open excess return versus SPY, using technical, volatility, macro, factor-residual and (optionally)
news-derived features. Output per stock: a score, a calibrated P(up), an expected excess return with a
conformal interval, and a within-day quantile label.

## Training and validation
- Data: daily bars (Yahoo, adjusted snapshot), FRED macro, Ken French factors (research only), RSS/GDELT news.
- Target: winsorised cross-sectional rank of the tradeable label (fill at t+1 open, exit at t+1+h open).
- Validation: purged, embargoed walk-forward (6 folds x 126 days). Stack weights for fold k use only folds < k.
  `CombinatorialPurgedCV` is available for path-level analysis.
- Hyper-parameters (when `--tune`) are searched on the first fold's training data only.
- Every configuration evaluated is counted in `runs/*.json` (`n_trials_total`) and fed to the deflated Sharpe.

## Measured performance (30 names, 2016-2026, price/macro features, h=5d)
| | rank IC | HAC t | gross Sharpe | net Sharpe (95% CI) |
|---|---|---|---|---|
| 12-1 momentum baseline | 0.025 | 1.3 | 0.34 | 0.30 (-0.33, 0.92) |
| LightGBM | 0.000 | 0.0 | -0.32 | -0.42 (-1.50, 0.61) |
| Ridge | -0.006 | -0.4 | -0.14 | -0.25 (-1.11, 0.79) |
| Stack | -0.003 | -0.2 | -0.17 | -0.26 (-1.24, 0.68) |

Net = after commission, half-spread, square-root impact and 30 bp/yr short borrow at USD 1M AUM, 5-day
rebalance, long/short top/bottom quintile. White's Reality Check p = 0.24; PBO = 0.50 (the in-sample winner is a
coin flip out of sample). Conclusion: **no tradeable edge was found**. This is the expected result for liquid
large caps with public price features.

## Intended use
Studying how to measure a signal honestly, and as a template for testing richer data (news, filings).
The portfolio layer scales alpha by walk-forward evidence (`evidence_scale`): with IC t <= 0 it holds nothing.

## Not intended for
Real-money trading, advice to third parties, small caps, intraday horizons, or any universe other than the one
tested without re-running the full validation.

## Known limitations
- Survivorship bias: the universe is a static list of today's large caps.
- Yahoo adjusted prices are restated snapshots, not point-in-time vendor data.
- News features have not been evaluated on a meaningful history; RSS history is shallow and GDELT covers about
  3 months of headlines. LLM extractions are subject to look-ahead contamination (use `mask_entities` ablations).
- Overlapping 5-day labels make naive hit-rate CIs optimistic; trust the HAC t-stat.
- Impact model is a generic square-root law, not calibrated to executions. Paper fills use next-open prices.
- The LSTM is evaluated in `train --lstm` but not persisted in the serving bundle.
- The API has no authentication and is intended for localhost.

## Monitoring
`stockpred monitor` / the scheduled monitor job reports feature drift (PSI, KS) between training data and data
that arrived after training, live IC from the prediction ledger (HAC t), and a retrain recommendation (model age,
drift share, live IC significantly negative with enough observations).
