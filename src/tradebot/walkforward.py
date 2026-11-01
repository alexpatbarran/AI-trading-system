"""Walk-forward training with purging.

For the model that trades day d:
  * training rows come only from days before d (rolling window);
  * rows whose label would END on or after d are dropped ("purging").
    A 5-day label from day d-2 includes prices from d+2, so keeping it would
    leak the future into training. Intraday labels never cross the session,
    so purging is a no-op there.
"""

from __future__ import annotations

import pandas as pd

from tradebot.strategy.base import Strategy


class WalkForward:
    def __init__(
        self,
        feats: pd.DataFrame,
        strategy: Strategy,
        required: list[str],
        train_days: int,
        min_train_days: int,
        retrain_every_days: int,
        min_train_samples: int,
    ):
        self.feats = feats
        self.strategy = strategy
        self.required = list(required)
        self.train_days = train_days
        self.min_train_days = min_train_days
        self.retrain_every_days = retrain_every_days
        self.min_train_samples = min_train_samples
        self.days = sorted(feats["session_date"].unique())
        self.last_fit: int | None = None
        self.fits = 0

    def training_rows(self, di: int) -> pd.DataFrame:
        day = self.days[di]
        window = self.days[max(0, di - self.train_days) : di]
        train = self.feats[self.feats["session_date"].isin(window)].dropna(subset=self.required + ["fwd_ret"])
        if "label_end_date" in train.columns:
            train = train[train["label_end_date"] < day]
        return train

    def step(self, di: int) -> bool:
        """Retrain if due; return whether the strategy can trade day `di`."""
        due = self.last_fit is None or di - self.last_fit >= self.retrain_every_days
        if due and di >= self.min_train_days:
            train = self.training_rows(di)
            if train["session_date"].nunique() >= self.min_train_days and len(train) >= self.min_train_samples:
                if self.strategy.fit(train):
                    self.last_fit = di
                    self.fits += 1
        return self.strategy.fitted
