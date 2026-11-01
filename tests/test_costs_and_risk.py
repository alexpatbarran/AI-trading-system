from datetime import datetime, timedelta, timezone

import pytest

from tradebot.config import CostConfig, RiskConfig
from tradebot.costs import CostModel
from tradebot.domain import Market, Position, Side, Signal
from tradebot.risk import RiskManager

NOW = datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)  # 11:00 New York


def make_risk(**overrides) -> RiskManager:
    cfg = RiskConfig(**overrides)
    costs = CostModel(CostConfig(spread_bps=2, slippage_bps=3))
    rm = RiskManager(cfg, Market.US, costs, horizon_bars=6, bar_minutes=5)
    rm.start_day(10_000)
    return rm


def sig(sym="AAA", er=0.01, p=0.6, price=100.0, ts=NOW):
    return Signal(Market.US, sym, ts, price, p, er)


def test_cost_model():
    c = CostModel(CostConfig(commission_bps=8, min_commission=6, spread_bps=6, slippage_bps=5))
    assert c.fill_price(Side.BUY, 100) == pytest.approx(100.08)
    assert c.fill_price(Side.SELL, 100) == pytest.approx(99.92)
    assert c.commission(10, 50) == 6  # minimum applies
    # $500 notional: 16 bps of spread/slippage + 2 x $6 commission (240 bps)
    assert c.round_trip_bps(10, 50) == pytest.approx(16 + 240)


def test_rejects_edge_below_costs():
    rm = make_risk()
    assert rm.entry_orders([sig(er=0.0008)], {}, 10_000, 10_000, NOW) == []  # 8 bps < 8 + 5
    assert rm.rejections["edge_below_costs"] == 1
    assert len(rm.entry_orders([sig(er=0.01)], {}, 10_000, 10_000, NOW)) == 1


def test_position_sizing_and_limits():
    rm = make_risk(max_position_pct=0.1, max_positions=2)
    orders = rm.entry_orders([sig("A"), sig("B"), sig("C")], {}, 10_000, 10_000, NOW)
    assert [o.qty for o in orders] == [10, 10]  # 10% of 10k at $100
    assert len(orders) == 2


def test_no_entries_near_close_and_flatten():
    rm = make_risk()
    late = datetime(2026, 7, 15, 19, 45, tzinfo=timezone.utc)  # 15:45 New York
    assert rm.entry_orders([sig()], {}, 10_000, 10_000, late) == []
    pos = {"A": Position(Market.US, "A", 10, 100, late - timedelta(minutes=5), 100)}
    exits = rm.exit_orders(pos, {"A": 100.0}, datetime(2026, 7, 15, 19, 52, tzinfo=timezone.utc), 10_000)
    assert exits[0].reason == "flatten_before_close"


def test_stop_take_profit_and_time_exit():
    rm = make_risk(stop_loss_pct=0.01, take_profit_pct=0.02)
    pos = {"A": Position(Market.US, "A", 10, 100, NOW, 100)}
    assert rm.exit_orders(pos, {"A": 98.9}, NOW + timedelta(minutes=5), 10_000)[0].reason == "stop_loss"
    assert rm.exit_orders(pos, {"A": 102.1}, NOW + timedelta(minutes=5), 10_000)[0].reason == "take_profit"
    assert rm.exit_orders(pos, {"A": 100.2}, NOW + timedelta(minutes=30), 10_000)[0].reason == "time_exit"
    assert rm.exit_orders(pos, {"A": 100.2}, NOW + timedelta(minutes=10), 10_000) == []


def test_daily_loss_kill_switch():
    rm = make_risk(daily_loss_limit_pct=0.02)
    pos = {"A": Position(Market.US, "A", 10, 100, NOW, 100)}
    exits = rm.exit_orders(pos, {"A": 100.0}, NOW, equity=9_790)
    assert exits[0].reason == "daily_loss_limit"
    assert rm.entry_orders([sig()], {}, 9_790, 9_790, NOW) == []
