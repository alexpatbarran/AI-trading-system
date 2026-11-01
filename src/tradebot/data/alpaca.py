"""US market data from Alpaca (pip install 'tradebot[alpaca]')."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from tradebot.data.base import DataProvider, normalise_bars
from tradebot.domain import Market


class AlpacaData(DataProvider):
    market = Market.US

    def __init__(self, feed: str = "iex", api_key: str | None = None, secret_key: str | None = None):
        from alpaca.data.historical import StockHistoricalDataClient

        self.feed = feed
        self.client = StockHistoricalDataClient(
            api_key or os.environ["ALPACA_API_KEY"],
            secret_key or os.environ["ALPACA_SECRET_KEY"],
        )

    def get_bars(self, symbols: list[str], start: datetime, end: datetime, bar_minutes: int) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        req = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed(self.feed),
            adjustment=Adjustment.SPLIT,
        )
        bars = self.client.get_stock_bars(req)
        df = bars.df.reset_index() if not bars.df.empty else pd.DataFrame()
        return normalise_bars(df)
