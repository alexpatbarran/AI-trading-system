"""ASX market data from Interactive Brokers.

Needs an ASX market data subscription on your IBKR account. IBKR applies
pacing limits to historical requests, hence the delay between symbols.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime

import pandas as pd

from tradebot.data.base import DataProvider, normalise_bars
from tradebot.domain import Market

log = logging.getLogger("tradebot.data.ibkr")

DAILY = 1440
BAR_SIZES = {1: "1 min", 2: "2 mins", 5: "5 mins", 15: "15 mins", 30: "30 mins", 60: "1 hour", DAILY: "1 day"}


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
        daily = bar_minutes == DAILY
        days = max(1, math.ceil((end - start).total_seconds() / 86400))
        duration = f"{math.ceil(days / 365)} Y" if days > 365 else f"{days} D"
        frames = []
        for sym in symbols:
            try:
                contract = Stock(sym, self.exchange, self.currency)
                self.ib.qualifyContracts(contract)
                bars = self.ib.reqHistoricalData(
                    contract,
                    # ADJUSTED_LAST (dividend/split adjusted) requires an empty end time
                    endDateTime="" if daily else end,
                    durationStr=duration,
                    barSizeSetting=BAR_SIZES[bar_minutes],
                    whatToShow="ADJUSTED_LAST" if daily else "TRADES",
                    useRTH=True,
                    formatDate=2,  # UTC timestamps
                )
            except Exception as exc:
                log.warning("Skipping %s: %s", sym, exc)
                continue
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
