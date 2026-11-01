"""Feature engineering and labels.

Rule: a feature at bar t may only use data available at the CLOSE of bar t.
Labels (fwd_ret) look forward by design and must never be used as inputs.
tests/test_features.py checks the no-look-ahead rule by truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.domain import Market
from tradebot.markets import SESSIONS

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "vol_12",
    "rel_volume",
    "range_pct",
    "close_loc",
    "dist_vwap",
    "rsi_14",
    "mins_since_open",
    "market_ret_6",
    "excess_ret_6",
]


def _roll(g, col: str, window: int, fn: str, min_periods: int | None = None, shift: int = 0) -> pd.Series:
    def f(s: pd.Series) -> pd.Series:
        s = s.shift(shift) if shift else s
        return getattr(s.rolling(window, min_periods=min_periods or window), fn)()

    return g[col].transform(f)


def add_features(bars: pd.DataFrame, market: Market) -> pd.DataFrame:
    s = SESSIONS[market]
    df = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    local = df["timestamp"].dt.tz_convert(s.tz)
    df["session_date"] = local.dt.date
    g = df.groupby("symbol", sort=False)

    for n in (1, 3, 6, 12):
        df[f"ret_{n}"] = df["close"] / g["close"].shift(n) - 1.0
    df["vol_12"] = _roll(g, "ret_1", 12, "std", min_periods=6)

    # Relative volume vs the PREVIOUS 48 bars (shifted so the current bar isn't in its own baseline)
    df["rel_volume"] = df["volume"] / _roll(g, "volume", 48, "mean", min_periods=12, shift=1)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["avg_dollar_volume"] = _roll(g, "dollar_volume", 48, "mean", min_periods=12)

    bar_range = df["high"] - df["low"]
    df["range_pct"] = bar_range / df["close"]
    df["close_loc"] = ((df["close"] - df["low"]) / bar_range.replace(0, np.nan)).fillna(0.5)

    # Session VWAP (resets each day)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_pv"] = typical * df["volume"]
    sg = df.groupby(["symbol", "session_date"], sort=False)
    vwap = sg["_pv"].cumsum() / sg["volume"].cumsum()
    df["dist_vwap"] = df["close"] / vwap - 1.0

    # RSI(14), simple-average version
    delta = g["close"].diff()
    df["_gain"] = delta.clip(lower=0)
    df["_loss"] = -delta.clip(upper=0)
    g = df.groupby("symbol", sort=False)
    avg_gain = _roll(g, "_gain", 14, "mean")
    avg_loss = _roll(g, "_loss", 14, "mean")
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(100.0).where(avg_gain.notna())

    open_min = s.open.hour * 60 + s.open.minute
    df["mins_since_open"] = (local.dt.hour * 60 + local.dt.minute - open_min).astype(float)

    # Cross-sectional context: how the whole universe moved over the same window
    df["market_ret_6"] = df.groupby("timestamp")["ret_6"].transform("mean")
    df["excess_ret_6"] = df["ret_6"] - df["market_ret_6"]

    return df.drop(columns=["_pv", "_gain", "_loss"])


def add_labels(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """fwd_ret = return from the NEXT bar's open to the close `horizon_bars` bars ahead.

    This matches execution: a decision at bar t's close fills at bar t+1's open.
    Labels that would cross into the next session are NaN (we flatten every day).
    """
    df = df.sort_values(["symbol", "timestamp"]).copy()
    g = df.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    entry_date = g["session_date"].shift(-1)
    exit_close = g["close"].shift(-horizon_bars)
    exit_date = g["session_date"].shift(-horizon_bars)
    same_session = (entry_date == df["session_date"]) & (exit_date == df["session_date"])
    df["fwd_ret"] = (exit_close / entry - 1.0).where(same_session)
    return df
