"""Trade journal: records every signal, order and equity snapshot.

This is the "continuous evaluation" layer: it lets you compare what the
model predicted with what actually happened, live, before trusting it with
more capital.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from tradebot.domain import Order, Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT, market TEXT, symbol TEXT, price REAL, prob_up REAL, expected_return REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, ts TEXT, market TEXT, symbol TEXT, side TEXT, qty INTEGER, reason TEXT, mode TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT, market TEXT, equity REAL, cash REAL, positions INTEGER
);
CREATE TABLE IF NOT EXISTS entries (
    market TEXT, symbol TEXT, entry_time TEXT, PRIMARY KEY (market, symbol)
);
"""


class Journal:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def signals(self, signals: list[Signal]) -> None:
        self.conn.executemany(
            "INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
            [
                (s.timestamp.isoformat(), s.market.value, s.symbol, s.price, s.prob_up, s.expected_return, s.reason)
                for s in signals
            ],
        )
        self.conn.commit()

    def order(self, o: Order, mode: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?)",
            (o.id, o.timestamp.isoformat(), o.market.value, o.symbol, o.side.value, o.qty, o.reason, mode),
        )
        self.conn.commit()

    def equity(self, ts: datetime, market: str, equity: float, cash: float, positions: int) -> None:
        self.conn.execute("INSERT INTO equity VALUES (?,?,?,?,?)", (ts.isoformat(), market, equity, cash, positions))
        self.conn.commit()

    def set_entry(self, market: str, symbol: str, ts: datetime | None) -> None:
        if ts is None:
            self.conn.execute("DELETE FROM entries WHERE market=? AND symbol=?", (market, symbol))
        else:
            self.conn.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?)", (market, symbol, ts.isoformat()))
        self.conn.commit()

    def entry_times(self, market: str) -> dict[str, datetime]:
        rows = self.conn.execute("SELECT symbol, entry_time FROM entries WHERE market=?", (market,)).fetchall()
        return {sym: datetime.fromisoformat(ts) for sym, ts in rows}
