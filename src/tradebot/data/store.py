"""SQLite bar store so historical data is fetched once and reused."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradebot.data.base import normalise_bars
from tradebot.domain import Market

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    market TEXT NOT NULL,
    bar_minutes INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (market, bar_minutes, symbol, ts)
);
"""


class BarStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def save(self, market: Market, bar_minutes: int, bars: pd.DataFrame) -> int:
        if bars.empty:
            return 0
        rows = [
            (market.value, bar_minutes, r.symbol, r.timestamp.isoformat(), r.open, r.high, r.low, r.close, r.volume)
            for r in bars.itertuples(index=False)
        ]
        self.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load(
        self,
        market: Market,
        bar_minutes: int,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        q = "SELECT ts AS timestamp, symbol, open, high, low, close, volume FROM bars WHERE market=? AND bar_minutes=?"
        params: list = [market.value, bar_minutes]
        if symbols:
            q += f" AND symbol IN ({','.join('?' * len(symbols))})"
            params += list(symbols)
        df = pd.read_sql_query(q, self.conn, params=params)
        df = normalise_bars(df)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] < pd.Timestamp(end)]
        return df.reset_index(drop=True)
