"""ASX market data from Interactive Brokers.

Needs an ASX market data subscription on your IBKR account. IBKR applies
pacing limits to historical requests, hence the delay between symbols.
"""

from __future__ import annotations

import math
import time
from datetime import datetime

import pandas as pd

from tradebot.data.base import DataProvider, normalise_bars
from tradebot.domain import Market

BAR_SIZES = {1: "1 min", 2: "2 mins", 5: "5 mins", 15: "15 mins", 30: "30 mins", 60: "1 hour"}


class IBKRData(DataProvider):
    market = Market.ASX

    def __init__(self, ib, exchange: str = "ASX", currency: str = "AUD", pace_seconds: float = 0.5):
        self.ib = ib
        self.exchange = exchange
        self.currency = currency
        self.pace_seconds = pace_seconds

    def get_bars(self, symbols: list[str], start: datetime, end: datetime, bar_minutes: int) -> pd.DataFrame:
        from ib_async import Stock, util

        if bar_minutes not in BAR_SIZES:
            raise ValueError(f"Unsupported IBKR bar size: {bar_minutes} min")
        days = max(1, math.ceil((end - start).total_seconds() / 86400))
        frames = []
        for sym in symbols:
            contract = Stock(sym, self.exchange, self.currency)
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end,
                durationStr=f"{days} D",
                barSizeSetting=BAR_SIZES[bar_minutes],
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,  # UTC timestamps
            )
            df = util.df(bars)
            if df is None or df.empty:
                continue
            df = df.rename(columns={"date": "timestamp"})
            df["symbol"] = sym
            frames.append(df)
            time.sleep(self.pace_seconds)
        if not frames:
            return normalise_bars(pd.DataFrame())
        out = normalise_bars(pd.concat(frames, ignore_index=True))
        return out[out["timestamp"] >= pd.Timestamp(start)]
