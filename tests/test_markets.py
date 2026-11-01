from datetime import date, datetime, timezone

from tradebot.domain import Market
from tradebot.markets import is_open, minutes_to_close, next_open, session_bounds


def test_us_session_handles_dst():
    # New York is UTC-5 in January, UTC-4 in July
    o_jan, _ = session_bounds(Market.US, date(2026, 1, 15))
    o_jul, _ = session_bounds(Market.US, date(2026, 7, 15))
    assert (o_jan.hour, o_jan.minute) == (14, 30)
    assert (o_jul.hour, o_jul.minute) == (13, 30)


def test_asx_session_handles_dst():
    # Sydney is UTC+11 in January (AEDT), UTC+10 in July (AEST)
    o_jan, c_jan = session_bounds(Market.ASX, date(2026, 1, 15))
    o_jul, _ = session_bounds(Market.ASX, date(2026, 7, 15))
    assert (o_jan.day, o_jan.hour, o_jan.minute) == (14, 23, 10)
    assert (o_jul.day, o_jul.hour, o_jul.minute) == (15, 0, 10)
    assert c_jan.hour == 5


def test_is_open_and_weekends():
    assert is_open(Market.US, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc))
    assert not is_open(Market.US, datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc))  # Saturday
    assert not is_open(Market.US, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc), {date(2026, 7, 15)})


def test_minutes_to_close_and_next_open():
    now = datetime(2026, 7, 15, 19, 30, tzinfo=timezone.utc)  # 15:30 New York
    assert minutes_to_close(Market.US, now) == 30
    nxt = next_open(Market.US, datetime(2026, 7, 17, 21, 0, tzinfo=timezone.utc))  # Friday evening
    assert nxt.date() == date(2026, 7, 20)
