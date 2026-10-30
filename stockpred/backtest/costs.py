"""Transaction-cost model: commission + half-spread + square-root market impact + short borrow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5  # per unit traded notional
    half_spread_bps: float = 2.0  # large-cap default; widen for small caps
    impact_coef: float = 0.1  # sqrt-law: cost = coef * daily_vol * sqrt(trade / ADV)
    borrow_bps_annual: float = 30.0  # general-collateral borrow on short notional

    def cost_frac(self, trade_dollars, adv_dollars, daily_vol):
        """Cost as a fraction of trade notional, elementwise (arrays or scalars). Participation is capped at
        100% of ADV; unknown ADV counts as zero impact (the fixed spread and commission still apply)."""
        td, adv, vol = (np.asarray(x, dtype=float) for x in (trade_dollars, adv_dollars, daily_vol))
        with np.errstate(divide="ignore", invalid="ignore"):
            part = np.where(adv > 0, np.minimum(np.abs(td) / adv, 1.0), 0.0)
        impact = self.impact_coef * np.nan_to_num(vol) * np.sqrt(np.nan_to_num(part))
        return (self.commission_bps + self.half_spread_bps) / 1e4 + impact

    def trade_cost_frac(self, trade_dollars: pd.DataFrame, adv_dollars: pd.DataFrame,
                        daily_vol: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.cost_frac(trade_dollars, adv_dollars, daily_vol), index=trade_dollars.index,
                            columns=trade_dollars.columns)

    def borrow_daily(self) -> float:
        return self.borrow_bps_annual / 1e4 / 252


ZERO_COST = CostModel(0.0, 0.0, 0.0, 0.0)
