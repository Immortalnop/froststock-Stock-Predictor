"""Purged / embargoed time-series splitters over a (date, symbol) panel.

A row at date t has a label that spans trading days t+1 .. t+1+h. Any training row whose
label window overlaps the test period must be purged; `embargo` adds a further safety gap
for serial correlation in features.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def _dates(index: pd.MultiIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    d = index.get_level_values("date")
    return d.to_numpy(), pd.DatetimeIndex(sorted(d.unique()))


class PurgedWalkForward:
    """Expanding (or rolling) window; train strictly ends `horizon+1+embargo` days before test."""

    def __init__(self, n_splits: int = 8, test_days: int = 126, min_train_days: int = 504,
                 horizon: int = 5, embargo: int = 2, max_train_days: int | None = None):
        self.n_splits, self.test_days, self.min_train_days = n_splits, test_days, min_train_days
        self.horizon, self.embargo, self.max_train_days = horizon, embargo, max_train_days

    @property
    def gap(self) -> int:
        return self.horizon + 1 + self.embargo

    def split(self, index: pd.MultiIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        d, ud = _dates(index)
        n = len(ud)
        need = self.min_train_days + self.gap + self.test_days
        if n < need:
            raise ValueError(f"need >= {need} dates, have {n}")
        # last test block ends at the final date; blocks tile backwards
        starts = [n - self.test_days * (self.n_splits - k) for k in range(self.n_splits)]
        starts = [s for s in starts if s - self.gap >= self.min_train_days]
        for s in starts:
            e = min(s + self.test_days, n)
            train_end = s - self.gap  # exclusive index into ud
            train_start = 0 if self.max_train_days is None else max(0, train_end - self.max_train_days)
            tr = (d >= ud[train_start]) & (d < ud[train_end])
            te = (d >= ud[s]) & (d <= ud[e - 1])
            yield np.flatnonzero(tr), np.flatnonzero(te)


class PurgedKFold:
    """K contiguous test blocks; training rows near either edge are purged/embargoed."""

    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo: int = 2):
        self.n_splits, self.horizon, self.embargo = n_splits, horizon, embargo

    def split(self, index: pd.MultiIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        d, ud = _dates(index)
        n = len(ud)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        for k in range(self.n_splits):
            a, b = bounds[k], bounds[k + 1]
            te = (d >= ud[a]) & (d <= ud[b - 1])
            # labels of rows just before the block reach into it -> purge them
            lo = max(0, a - (self.horizon + 1))
            purge = (d >= ud[lo]) & (d < ud[a])
            # embargo: drop rows right after the block (serial correlation in features)
            eb = min(n, b + self.embargo)
            embargo = (d >= ud[b]) & (d <= ud[eb - 1]) if b < n else np.zeros(len(d), bool)
            yield np.flatnonzero(~te & ~purge & ~embargo), np.flatnonzero(te)


class CombinatorialPurgedCV:
    """CPCV (Lopez de Prado): N contiguous groups, every combination of k as test.

    Yields C(N,k) splits; each group is tested in C(N-1,k-1) of them, so the backtest paths
    can be reassembled into C(N-1,k-1) distinct OOS paths (see `n_paths`). Training
    rows whose label window reaches into a test group, or that fall within `embargo` days after
    one, are removed.
    """

    def __init__(self, n_groups: int = 6, n_test: int = 2, horizon: int = 5, embargo: int = 2):
        if not 0 < n_test < n_groups:
            raise ValueError("need 0 < n_test < n_groups")
        self.n_groups, self.n_test, self.horizon, self.embargo = n_groups, n_test, horizon, embargo

    @property
    def n_splits(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test)

    @property
    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups - 1, self.n_test - 1)

    def split(self, index: pd.MultiIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from itertools import combinations

        d, ud = _dates(index)
        n = len(ud)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        for combo in combinations(range(self.n_groups), self.n_test):
            te = np.zeros(len(d), bool)
            excl = np.zeros(len(d), bool)
            for g in combo:
                a, b = bounds[g], bounds[g + 1]
                te |= (d >= ud[a]) & (d <= ud[b - 1])
                lo = max(0, a - (self.horizon + 1))
                excl |= (d >= ud[lo]) & (d <= ud[min(n, b + self.embargo) - 1])
            yield np.flatnonzero(~excl), np.flatnonzero(te)
