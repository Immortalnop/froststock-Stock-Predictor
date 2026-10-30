import numpy as np
import pandas as pd
import pytest

from stockpred.config.universe import Security, Universe


@pytest.fixture
def universe():
    return Universe([
        Security("AAPL", "Apple", "Technology", ("iPhone", "Tim Cook")),
        Security("MSFT", "Microsoft", "Technology", ("Azure",)),
        Security("XOM", "Exxon Mobil", "Energy", ("Exxon",)),
        Security("V", "Visa", "Financials"),
    ])


def make_prices(n=900, seed=0, start="2018-01-02", drift=0.0003, vol=0.015):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol_ = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "adj_close": close, "volume": vol_}, index=idx)


@pytest.fixture
def px():
    return make_prices()
