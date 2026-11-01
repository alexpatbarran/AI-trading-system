"""Rule-based score, calibrated empirically on the training window.

The score is a fixed rule. What gets learned is the mapping from score bucket
to historical forward return and hit rate. If the rule doesn't work in the
training window, the buckets say so and the risk gates won't trade it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from tradebot.strategy.base import Strategy


class RankStrategy(Strategy):
    name = "rank"

    def __init__(
        self,
        score_fn: Callable[[pd.DataFrame], pd.Series],
        name: str | None = None,
        n_buckets: int = 10,
        min_bucket_size: int = 30,
    ):
        super().__init__()
        self.score_fn = score_fn
        if name:
            self.name = name
        self.n_buckets = n_buckets
        self.min_bucket_size = min_bucket_size
        self.edges: np.ndarray | None = None
        self.bucket_mean: np.ndarray | None = None
        self.bucket_prob: np.ndarray | None = None

    def score(self, df: pd.DataFrame) -> pd.Series:
        return self.score_fn(df)

    def fit(self, train: pd.DataFrame) -> bool:
        sc = self.score(train)
        y = train["fwd_ret"]
        mask = sc.notna() & y.notna() & np.isfinite(sc)
        sc, y = sc[mask].to_numpy(), y[mask].to_numpy()
        if len(sc) < self.n_buckets * self.min_bucket_size:
            return False
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        edges = np.unique(np.quantile(sc, qs))
        idx = np.searchsorted(edges, sc)
        n = len(edges) + 1
        means, probs = np.zeros(n), np.full(n, 0.5)
        for b in range(n):
            yb = y[idx == b]
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
