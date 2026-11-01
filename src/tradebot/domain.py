"""Core domain types shared across the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

# Bars are passed around as pandas DataFrames in "long" format with columns:
#   timestamp (tz-aware UTC, bar START time), symbol, open, high, low, close, volume
BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


class Market(str, Enum):
    US = "US"
    ASX = "ASX"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Signal:
    """A model's view on one symbol at one decision time."""

    market: Market
    symbol: str
    timestamp: datetime
    price: float
    prob_up: float
    expected_return: float  # fraction over the horizon, BEFORE costs
    reason: str = ""


@dataclass
class Order:
    market: Market
    symbol: str
    side: Side
    qty: int
    timestamp: datetime
    reason: str = ""
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    spread_bps: float | None = None  # per-stock spread estimate (simulation only)


@dataclass(frozen=True)
class Fill:
    order_id: str
    market: Market
    symbol: str
    side: Side
    qty: int
    price: float  # includes spread and slippage
    commission: float
    timestamp: datetime
    reason: str = ""


@dataclass
class Position:
    market: Market
    symbol: str
    qty: int
    avg_price: float
    entry_time: datetime
    last_price: float = 0.0
    entry_costs: float = 0.0  # commission + spread/slippage paid on entry

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price or self.avg_price)

    @property
    def unrealised_return(self) -> float:
        if not self.last_price or not self.avg_price:
            return 0.0
        return self.last_price / self.avg_price - 1.0
