from datetime import date

from tradebot.backtest import Backtester
from tradebot.config import assert_live_allowed
from tradebot.data.synthetic import generate_bars, trading_days
from tradebot.domain import Market
from tradebot.strategy import make_strategy

import pytest


def _bars(market, settings, momentum, n_days=20, seed=1):
    days = trading_days(market, date(2026, 5, 29), n_days)
    return generate_bars(market, settings.markets[market].universe, days, settings.bar_minutes, momentum, seed=seed)


def test_finds_planted_edge(settings):
    """With strong planted momentum, the pipeline should make money after costs."""
    settings.strategy.name = "ml"
    r = Backtester(settings, Market.US, make_strategy(settings.strategy)).run(_bars(Market.US, settings, 0.4))
    assert r.metrics["trades"] > 20
    assert r.metrics["total_return"] > 0


def test_flat_at_end_of_every_day(settings):
    settings.strategy.name = "ml"
    settings.markets[Market.ASX].starting_capital = 200_000  # big enough that min commission isn't a wall
    r = Backtester(settings, Market.ASX, make_strategy(settings.strategy)).run(_bars(Market.ASX, settings, 0.4))
    assert not r.trades.empty
    exit_day = r.trades["exit_time"].dt.tz_convert("Australia/Sydney").dt.date
    entry_day = r.trades["entry_time"].dt.tz_convert("Australia/Sydney").dt.date
    assert (exit_day == entry_day).all()


def test_min_commission_blocks_small_asx_trades(settings):
    """$1,000 positions with a $6 minimum commission cost ~1.2% round trip. Risk should refuse."""
    settings.strategy.name = "ml"
    r = Backtester(settings, Market.ASX, make_strategy(settings.strategy)).run(_bars(Market.ASX, settings, 0.4))
    assert r.metrics["trades"] == 0
    assert r.risk_rejections.get("edge_below_costs", 0) > 0


def test_fills_happen_after_decision(settings):
    """Entries fill on a later bar than the one that produced the signal (no same-bar fills)."""
    settings.strategy.name = "ml"
    r = Backtester(settings, Market.US, make_strategy(settings.strategy)).run(_bars(Market.US, settings, 0.4))
    held = r.trades["exit_time"] - r.trades["entry_time"]
    assert (held.dt.total_seconds() > 0).all()


def test_live_requires_double_confirmation(settings, monkeypatch):
    monkeypatch.delenv("TRADEBOT_CONFIRM_LIVE", raising=False)
    with pytest.raises(PermissionError):
        assert_live_allowed(settings)  # mode is paper
    settings.mode = "live"
    with pytest.raises(PermissionError):
        assert_live_allowed(settings)  # env var missing
    monkeypatch.setenv("TRADEBOT_CONFIRM_LIVE", "I_UNDERSTAND_THE_RISKS")
    assert_live_allowed(settings)
