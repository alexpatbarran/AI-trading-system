"""US market data from Alpaca (pip install 'tradebot[alpaca]')."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from tradebot.data.base import DataProvider, normalise_bars
from tradebot.domain import Market

log = logging.getLogger("tradebot.data.alpaca")

DAILY = 1440
CHUNK = 50


class AlpacaData(DataProvider):
    market = Market.US

    def __init__(
        self,
        feed: str = "iex",
        daily_feed: str = "sip",
        api_key: str | None = None,
        secret_key: str | None = None,
    ):
        from alpaca.data.historical import StockHistoricalDataClient

        self.feed = feed
        self.daily_feed = daily_feed
        self.client = StockHistoricalDataClient(
            api_key or os.environ["ALPACA_API_KEY"],
            secret_key or os.environ["ALPACA_SECRET_KEY"],
        )

    def get_bars(self, symbols: list[str], start: datetime, end: datetime, bar_minutes: int) -> pd.DataFrame:
        feed = self.daily_feed if bar_minutes == DAILY else self.feed
        frames = []
        for i in range(0, len(symbols), CHUNK):
            frames.append(self._fetch_chunk(list(symbols[i : i + CHUNK]), start, end, bar_minutes, feed))
        frames = [f for f in frames if not f.empty]
        return normalise_bars(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())

    def _fetch_chunk(self, symbols, start, end, bar_minutes, feed) -> pd.DataFrame:
        try:
            return self._request(symbols, start, end, bar_minutes, feed)
        except Exception as exc:
            msg = str(exc).lower()
            if feed == "sip" and ("subscription" in msg or "permit" in msg):
                log.warning("SIP data not available on this plan (%s); falling back to IEX", exc)
                if bar_minutes == DAILY:
                    self.daily_feed = "iex"
                else:
                    self.feed = "iex"
                return self._fetch_chunk(symbols, start, end, bar_minutes, "iex")
            if len(symbols) == 1:
                log.warning("Skipping %s: %s", symbols[0], exc)
                return pd.DataFrame()
            log.warning("Chunk failed (%s); retrying symbols one at a time", exc)
            parts = [self._fetch_chunk([s], start, end, bar_minutes, feed) for s in symbols]
            parts = [p for p in parts if not p.empty]
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _request(self, symbols, start, end, bar_minutes, feed) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if feed == "sip":  # free plans can't query the most recent 15 minutes of SIP data
            end = min(end, datetime.now(timezone.utc) - timedelta(minutes=16))
        daily = bar_minutes == DAILY
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day if daily else TimeFrame(bar_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed(feed),
            # daily: adjust for dividends too, so ex-dividend drops aren't mistaken for losses
            adjustment=Adjustment.ALL if daily else Adjustment.SPLIT,
        )
        bars = self.client.get_stock_bars(req)
        return bars.df.reset_index() if not bars.df.empty else pd.DataFrame()
