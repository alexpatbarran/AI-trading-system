"""Market data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

from tradebot.domain import BAR_COLUMNS, Market


class DataProvider(ABC):
    market: Market

    @abstractmethod
    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, bar_minutes: int
    ) -> pd.DataFrame:
        """Return bars in long format (see domain.BAR_COLUMNS), timestamps = bar start, UTC."""


def normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce provider output to the standard bar schema."""
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    out = df[BAR_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = out[col].astype(float)
    return (
        out.dropna(subset=["close"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def drop_incomplete_bars(df: pd.DataFrame, now: datetime, bar_minutes: int) -> pd.DataFrame:
    """Remove bars that haven't closed yet. Trading on a forming bar is a classic look-ahead bug."""
    if df.empty:
        return df
    closes = df["timestamp"] + timedelta(minutes=bar_minutes)
    return df[closes <= pd.Timestamp(now)].reset_index(drop=True)
