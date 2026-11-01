"""Synthetic intraday bars for offline development and tests.

By default the generated prices have NO predictable structure, so any
strategy that "profits" on them is fooling itself. Set `momentum` > 0 to
plant autocorrelation and confirm the pipeline can find a real edge.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from tradebot.data.base import DataProvider, normalise_bars
from tradebot.domain import Market
from tradebot.markets import is_trading_day, session_bounds


def trading_days(market: Market, end: date, n_days: int, holidays: set[date] | None = None) -> list[date]:
    days: list[date] = []
    d = end
    while len(days) < n_days:
        if is_trading_day(market, d, holidays):
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def generate_bars(
    market: Market,
    symbols: list[str],
    days: list[date],
    bar_minutes: int = 5,
    momentum: float = 0.0,
    vol_per_bar: float = 0.002,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, sym in enumerate(symbols):
        price = float(rng.uniform(20, 200))
        base_volume = float(rng.uniform(20_000, 200_000))
        prev_ret = 0.0
        rows = []
        for d in days:
            o, c = session_bounds(market, d)
            # overnight gap
            price *= float(np.exp(rng.normal(0, vol_per_bar * 3)))
            ts = o
            while ts + timedelta(minutes=bar_minutes) <= c:
                shock = rng.normal(0, vol_per_bar)
                ret = momentum * prev_ret + shock
                open_ = price
                close = open_ * float(np.exp(ret))
                wiggle = abs(rng.normal(0, vol_per_bar / 2))
                high = max(open_, close) * (1 + wiggle)
                low = min(open_, close) * (1 - wiggle)
                volume = base_volume * float(np.exp(rng.normal(0, 0.4))) * (1 + 3 * abs(ret) / vol_per_bar)
                rows.append((ts, sym, open_, high, low, close, volume))
                price, prev_ret = close, ret
                ts += timedelta(minutes=bar_minutes)
        frames.append(pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]))
    return normalise_bars(pd.concat(frames, ignore_index=True))


def generate_daily_bars(
    market: Market,
    symbols: list[str],
    days: list[date],
    momentum: float = 0.0,
    seed: int = 0,
    drift: float = 0.0003,
) -> pd.DataFrame:
    """Daily bars with a common market factor plus stock-specific noise.

    Note: with a positive market `drift`, higher-beta stocks earn slightly more
    on average. That's a real (tiny) cross-sectional effect a model can pick up.

    `momentum` plants autocorrelation in each stock's idiosyncratic return
    against its own average over the previous 5 days. Negative = short-term
    reversal, positive = continuation. 0 = nothing to find.
    """
    rng = np.random.default_rng(seed)
    n = len(days)
    market_ret = rng.normal(drift, 0.01, n)
    frames = []
    for sym in symbols:
        beta = rng.uniform(0.7, 1.3)
        idio_vol = rng.uniform(0.01, 0.025)
        base_volume = rng.uniform(2e6, 2e7)
        idio = np.zeros(n)
        price = float(rng.uniform(20, 300))
        rows = []
        for t, d in enumerate(days):
            past = idio[max(0, t - 5) : t]
            idio[t] = momentum * (past.mean() if len(past) else 0.0) + rng.normal(0, idio_vol)
            ret = beta * market_ret[t] + idio[t]
            open_ = price * float(np.exp(rng.normal(0, 0.003)))
            close = price * float(np.exp(ret))
            wiggle = abs(rng.normal(0, 0.006))
            high = max(open_, close) * (1 + wiggle)
            low = min(open_, close) * (1 - wiggle)
            volume = base_volume * float(np.exp(rng.normal(0, 0.3))) * (1 + 20 * abs(ret))
            o, _ = session_bounds(market, d)
            rows.append((o, sym, open_, high, low, close, volume))
            price = close
        frames.append(pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]))
    return normalise_bars(pd.concat(frames, ignore_index=True))


class SyntheticData(DataProvider):
    def __init__(self, market: Market, momentum: float = 0.0, seed: int = 0):
        self.market = market
        self.momentum = momentum
        self.seed = seed

    def get_bars(self, symbols: list[str], start: datetime, end: datetime, bar_minutes: int) -> pd.DataFrame:
        days = []
        d = start.date()
        while d <= end.date():
            if is_trading_day(self.market, d):
                days.append(d)
            d += timedelta(days=1)
        df = generate_bars(self.market, symbols, days, bar_minutes, self.momentum, seed=self.seed)
        return df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end))]
