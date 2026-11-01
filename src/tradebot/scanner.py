"""Market scanner: cut the universe down to liquid, active candidates."""

from __future__ import annotations

import pandas as pd

from tradebot.config import MarketConfig, ScannerConfig
from tradebot.features import FEATURES


def scan(snapshot: pd.DataFrame, market_cfg: MarketConfig, cfg: ScannerConfig) -> pd.DataFrame:
    """`snapshot` holds one row per symbol at a single timestamp, with features."""
    s = snapshot.dropna(subset=FEATURES)
    s = s[
        (s["close"] >= market_cfg.min_price)
        & (s["avg_dollar_volume"] >= market_cfg.min_bar_dollar_volume)
        & (s["rel_volume"] >= cfg.min_relative_volume)
    ]
    return s.sort_values("rel_volume", ascending=False).head(cfg.top_n)
