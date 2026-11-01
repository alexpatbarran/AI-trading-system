from datetime import date

import pandas as pd

from tradebot.data.synthetic import generate_bars, trading_days
from tradebot.domain import Market
from tradebot.markets import regular_session_only
from tradebot.research import diagnose, format_diagnosis
from tradebot.strategy import make_strategy


def test_regular_session_only_drops_extended_hours(us_bars):
    extra = us_bars.head(3).copy()
    extra["timestamp"] = pd.Timestamp("2026-03-20 12:00", tz="UTC")  # 08:00 New York, pre-market
    mixed = pd.concat([us_bars, extra], ignore_index=True)
    assert len(regular_session_only(mixed, Market.US)) == len(us_bars)


def test_diagnose_detects_planted_signal(settings):
    days = trading_days(Market.US, date(2026, 5, 29), 20)
    bars = generate_bars(Market.US, settings.markets[Market.US].universe, days, 5, momentum=0.4, seed=2)
    d = diagnose(settings, Market.US, make_strategy(settings.strategy), bars)
    assert d is not None
    assert d.ic_mean > 0 and d.ic_tstat > 3
    assert d.deciles["realised_bps"].iloc[-1] > d.deciles["realised_bps"].iloc[0]
    assert "verdict" in format_diagnosis(d)
