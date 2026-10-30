"""LSTM over a lookback window of per-symbol features (torch is an optional dependency).

Windows are built with causal per-symbol lags in the panel (`make_lagged`), so the model API
stays tabular and CV code is shared: columns are `<feature>__l<k>` with k=0 the current day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_lagged(panel: pd.DataFrame, cols: list[str], seq_len: int = 20) -> pd.DataFrame:
    """Lag-stack `cols` for k in 0..seq_len-1 using only past rows of the same symbol."""
    g = panel[cols].groupby(level="symbol")
    parts = {}
    for k in range(seq_len):
        shifted = panel[cols] if k == 0 else g.shift(k)
        for c in cols:
            parts[f"{c}__l{k}"] = shifted[c]
    return pd.DataFrame(parts, index=panel.index)


def _make_net(n_features: int, hidden: int):
    from torch import nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1]).squeeze(-1)

    return LSTMNet()


class LSTMRegressor:
    def __init__(self, base_cols: list[str], seq_len: int = 20, hidden: int = 32, epochs: int = 6,
                 lr: float = 2e-3, batch: int = 1024, weight_decay: float = 1e-4, seed: int = 7):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("LSTMRegressor needs torch: pip install torch") from exc
        self.base_cols, self.seq_len = list(base_cols), seq_len
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.batch, self.wd, self.seed = batch, weight_decay, seed
        self.cols_ = [f"{c}__l{k}" for k in range(seq_len - 1, -1, -1) for c in base_cols]  # oldest first
        self.net = None

    def _tensor(self, X: pd.DataFrame):
        import torch

        a = X[self.cols_].to_numpy(np.float32)
        a = np.clip((a - self.mu_) / self.sd_, -5, 5)
        a = np.nan_to_num(a, nan=0.0)
        return torch.from_numpy(a.reshape(len(a), self.seq_len, len(self.base_cols)))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMRegressor":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        m = y.notna().to_numpy()
        Xtr, ytr = X[m], y[m].to_numpy(np.float32)
        raw = Xtr[self.cols_].to_numpy(np.float32)
        self.mu_ = np.nanmean(raw, axis=0)
        self.sd_ = np.nanstd(raw, axis=0) + 1e-6
        T = self._tensor(Xtr)
        Y = torch.from_numpy(ytr)

        self.net = _make_net(len(self.base_cols), self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()
        n = len(T)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i : i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(T[idx]), Y[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            out = []
            T = self._tensor(X)
            for i in range(0, len(T), 4096):
                out.append(self.net(T[i : i + 4096]).numpy())
        return np.concatenate(out) if out else np.array([])
