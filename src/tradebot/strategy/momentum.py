"""Baseline: risk-adjusted short-term momentum, calibrated empirically.

The score itself is a fixed rule. What gets "learned" is the mapping from
score bucket -> historical forward return and hit rate. If momentum doesn't
work in the training window the buckets say so, and the risk manager won't
trade. Any ML model has to beat this to justify its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, n_buckets: int = 10, min_bucket_size: int = 30):
        super().__init__()
        self.n_buckets = n_buckets
        self.min_bucket_size = min_bucket_size
        self.edges: np.ndarray | None = None
        self.bucket_mean: np.ndarray | None = None
        self.bucket_prob: np.ndarray | None = None

    @staticmethod
    def score(df: pd.DataFrame) -> pd.Series:
        vol = df["vol_12"].replace(0, np.nan) * np.sqrt(6)
        return df["excess_ret_6"] / vol

    def fit(self, train: pd.DataFrame) -> bool:
        sc = self.score(train)
        y = train["fwd_ret"]
        mask = sc.notna() & y.notna() & np.isfinite(sc)
        sc, y = sc[mask], y[mask]
        if len(sc) < self.n_buckets * self.min_bucket_size:
            return False
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        edges = np.unique(np.quantile(sc, qs))
        idx = np.searchsorted(edges, sc.to_numpy())
        n = len(edges) + 1
        means, probs = np.zeros(n), np.full(n, 0.5)
        for b in range(n):
            yb = y.to_numpy()[idx == b]
            if len(yb) >= self.min_bucket_size:
                means[b] = yb.mean()
                probs[b] = (yb > 0).mean()
        self.edges, self.bucket_mean, self.bucket_prob = edges, means, probs
        self.fitted = True
        return True

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Strategy not fitted")
        sc = self.score(rows).fillna(0.0).to_numpy()
        idx = np.searchsorted(self.edges, sc)
        return pd.DataFrame(
            {"prob_up": self.bucket_prob[idx], "expected_return": self.bucket_mean[idx]},
            index=rows.index,
        )
