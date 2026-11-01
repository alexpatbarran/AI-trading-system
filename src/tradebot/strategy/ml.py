"""Gradient-boosted model predicting direction and size of the forward return.

Deliberately heavily regularised (shallow trees, large leaves): intraday
returns are mostly noise and an unconstrained model will memorise it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tradebot.features import FEATURES
from tradebot.strategy.base import Strategy


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, features: list[str] | None = None, seed: int = 0):
        super().__init__()
        self.features = features or FEATURES
        params = dict(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=seed,
        )
        self.clf = HistGradientBoostingClassifier(**params)
        self.reg = HistGradientBoostingRegressor(**params)

    def fit(self, train: pd.DataFrame) -> bool:
        data = train.dropna(subset=self.features + ["fwd_ret"])
        if len(data) < 1000:
            return False
        X = data[self.features].to_numpy()
        y = data["fwd_ret"].to_numpy()
        up = (y > 0).astype(int)
        if up.min() == up.max():
            return False
        lo, hi = np.quantile(y, [0.01, 0.99])  # tame fat tails for the regressor
        self.clf.fit(X, up)
        self.reg.fit(X, np.clip(y, lo, hi))
        self.fitted = True
        return True

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Strategy not fitted")
        X = rows[self.features].to_numpy()
        return pd.DataFrame(
            {"prob_up": self.clf.predict_proba(X)[:, 1], "expected_return": self.reg.predict(X)},
            index=rows.index,
        )
