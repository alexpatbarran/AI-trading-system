"""Intraday baseline: risk-adjusted short-term momentum, calibrated empirically.

Any ML model has to beat this to justify its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategy.rank import RankStrategy


def momentum_score(df: pd.DataFrame) -> pd.Series:
    vol = df["vol_12"].replace(0, np.nan) * np.sqrt(6)
    return df["excess_ret_6"] / vol


class MomentumStrategy(RankStrategy):
    name = "momentum"

    def __init__(self, n_buckets: int = 10, min_bucket_size: int = 30):
        super().__init__(momentum_score, n_buckets=n_buckets, min_bucket_size=min_bucket_size)
