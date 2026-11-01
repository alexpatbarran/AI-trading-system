"""Daily features and labels for swing trading.

Rule: features on day t use only data up to day t's CLOSE. The decision is
made after the close; the trade fills at day t+1's OPEN.

The features encode effects documented in academic research:
  * short-term reversal (last week's losers tend to bounce)    -> ret_5d, excess_ret_5d
  * medium-term momentum ("12-1": 12-month return skipping the last month) -> mom_12_1
  * 52-week-high proximity                                       -> dist_52w_high
  * trend, volatility and volume context
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.domain import Market
from tradebot.markets import SESSIONS

# Features that make sense to compare ACROSS stocks on the same day
CS_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "mom_12_1",
    "vol_21d",
    "vol_ratio",
    "dist_ma50",
    "dist_ma200",
    "dist_52w_high",
    "gap",
    "range_pct",
    "rsi_14",
]
MARKET_FEATURES = ["market_ret_5d", "market_ret_21d", "market_vol_21d"]
DAILY_FEATURES = CS_FEATURES + ["excess_ret_5d", "excess_ret_21d"] + MARKET_FEATURES
# What the ML model sees: cross-sectional percentile ranks (robust to regime shifts) + market context
DAILY_ML_FEATURES = [f"cs_{f}" for f in CS_FEATURES] + MARKET_FEATURES


def _roll(df: pd.DataFrame, col: str, window: int, fn: str, min_periods: int, shift: int = 0) -> pd.Series:
    def f(s: pd.Series) -> pd.Series:
        s = s.shift(shift) if shift else s
        return getattr(s.rolling(window, min_periods=min_periods), fn)()

    return df.groupby("symbol", sort=False)[col].transform(f)


def add_daily_features(bars: pd.DataFrame, market: Market) -> pd.DataFrame:
    tz = SESSIONS[market].tz
    df = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    df["session_date"] = df["timestamp"].dt.tz_convert(tz).dt.date
    g = df.groupby("symbol", sort=False)
    close = df["close"]

    df["ret_1d"] = close / g["close"].shift(1) - 1
    df["ret_5d"] = close / g["close"].shift(5) - 1
    df["ret_21d"] = close / g["close"].shift(21) - 1
    df["mom_12_1"] = g["close"].shift(21) / g["close"].shift(252) - 1
    df["vol_21d"] = _roll(df, "ret_1d", 21, "std", 15)
    df["vol_ratio"] = df["volume"] / _roll(df, "volume", 21, "mean", 15, shift=1)
    df["dist_ma50"] = close / _roll(df, "close", 50, "mean", 40) - 1
    df["dist_ma200"] = close / _roll(df, "close", 200, "mean", 150) - 1
    df["dist_52w_high"] = close / _roll(df, "high", 252, "max", 200) - 1
    df["gap"] = df["open"] / g["close"].shift(1) - 1
    df["range_pct"] = (df["high"] - df["low"]) / close

    delta = g["close"].diff()
    df["_gain"] = delta.clip(lower=0)
    df["_loss"] = -delta.clip(upper=0)
    avg_gain = _roll(df, "_gain", 14, "mean", 14)
    avg_loss = _roll(df, "_loss", 14, "mean", 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(100.0).where(avg_gain.notna())

    df["dollar_volume"] = close * df["volume"]
    df["avg_dollar_volume"] = _roll(df, "dollar_volume", 21, "mean", 15)

    by_day = df.groupby("session_date")
    df["market_ret_5d"] = by_day["ret_5d"].transform("mean")
    df["market_ret_21d"] = by_day["ret_21d"].transform("mean")
    df["market_vol_21d"] = by_day["vol_21d"].transform("mean")
    df["excess_ret_5d"] = df["ret_5d"] - df["market_ret_5d"]
    df["excess_ret_21d"] = df["ret_21d"] - df["market_ret_21d"]
    for f in CS_FEATURES:
        df[f"cs_{f}"] = by_day[f].rank(pct=True)
    # 1 = most traded stock that day (point-in-time size proxy)
    df["size_rank"] = by_day["avg_dollar_volume"].rank(ascending=False, method="first")

    df = df.drop(columns=["_gain", "_loss"])
    # float32 halves memory for broad universes (millions of rows)
    skip = {"open", "high", "low", "close", "volume"}
    floats = [c for c in df.columns if df[c].dtype == "float64" and c not in skip]
    df[floats] = df[floats].astype("float32")
    return df


def add_daily_labels(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """fwd_ret = open(t+1) -> close(t+horizon). label_end_date is used to purge training data."""
    df = df.sort_values(["symbol", "timestamp"]).copy()
    g = df.groupby("symbol", sort=False)
    df["fwd_ret"] = g["close"].shift(-horizon_days) / g["open"].shift(-1) - 1
    df["label_end_date"] = g["session_date"].shift(-horizon_days)
    return df
