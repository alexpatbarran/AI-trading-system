"""Exchange sessions for each market (regular trading hours only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from tradebot.domain import Market


@dataclass(frozen=True)
class Session:
    market: Market
    tz: ZoneInfo
    open: time
    close: time
    currency: str


SESSIONS: dict[Market, Session] = {
    Market.US: Session(Market.US, ZoneInfo("America/New_York"), time(9, 30), time(16, 0), "USD"),
    # ASX opens in a staggered auction 10:00-10:09 and closes with an auction
    # at ~16:10. We trade only continuous trading, from 10:10 to 16:00.
    Market.ASX: Session(Market.ASX, ZoneInfo("Australia/Sydney"), time(10, 10), time(16, 0), "AUD"),
}

# Holidays come from config (MarketConfig.holidays). Half days (e.g. US
# day after Thanksgiving) are NOT modelled yet - see README roadmap.


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc)


def local_date(market: Market, ts: datetime) -> date:
    return _as_utc(ts).astimezone(SESSIONS[market].tz).date()


def session_bounds(market: Market, day: date) -> tuple[datetime, datetime]:
    """(open, close) in UTC for a local trading date."""
    s = SESSIONS[market]
    o = datetime.combine(day, s.open, tzinfo=s.tz).astimezone(timezone.utc)
    c = datetime.combine(day, s.close, tzinfo=s.tz).astimezone(timezone.utc)
    return o, c


def is_trading_day(market: Market, day: date, holidays: set[date] | None = None) -> bool:
    return day.weekday() < 5 and day not in (holidays or set())


def is_open(market: Market, now: datetime, holidays: set[date] | None = None) -> bool:
    now = _as_utc(now)
    day = local_date(market, now)
    if not is_trading_day(market, day, holidays):
        return False
    o, c = session_bounds(market, day)
    return o <= now < c


def minutes_to_close(market: Market, now: datetime) -> float:
    now = _as_utc(now)
    _, c = session_bounds(market, local_date(market, now))
    return (c - now).total_seconds() / 60.0


def regular_session_only(bars, market: Market):
    """Drop pre-market, after-hours and auction bars (Alpaca returns extended hours by default)."""
    if bars.empty:
        return bars
    s = SESSIONS[market]
    local = bars["timestamp"].dt.tz_convert(s.tz)
    t = local.dt.time
    mask = (t >= s.open) & (t < s.close) & (local.dt.weekday < 5)
    return bars[mask].reset_index(drop=True)


def next_open(market: Market, now: datetime, holidays: set[date] | None = None) -> datetime:
    now = _as_utc(now)
    day = local_date(market, now)
    for _ in range(15):
        if is_trading_day(market, day, holidays):
            o, _ = session_bounds(market, day)
            if o > now:
                return o
        day += timedelta(days=1)
    raise RuntimeError("No trading day found in the next 15 days")
