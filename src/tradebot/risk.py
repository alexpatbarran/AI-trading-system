"""Risk management: every order goes through here. The model proposes, risk disposes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradebot.config import RiskConfig
from tradebot.costs import CostModel
from tradebot.domain import Market, Order, Position, Side, Signal
from tradebot.markets import minutes_to_close


@dataclass
class RiskManager:
    cfg: RiskConfig
    market: Market
    costs: CostModel
    horizon_bars: int
    bar_minutes: int
    min_prob_up: float = 0.5
    day_start_equity: float | None = None
    killed: bool = False
    rejections: dict[str, int] = field(default_factory=dict)

    # --- day lifecycle -------------------------------------------------
    def start_day(self, equity: float) -> None:
        self.day_start_equity = equity
        self.killed = False

    def update_kill_switch(self, equity: float) -> bool:
        if self.day_start_equity and not self.killed:
            if equity / self.day_start_equity - 1.0 <= -self.cfg.daily_loss_limit_pct:
                self.killed = True
        return self.killed

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    # --- exits -----------------------------------------------------------
    def exit_orders(
        self,
        positions: dict[str, Position],
        prices: dict[str, float],
        now: datetime,
        equity: float,
        pending: set[str] | None = None,
    ) -> list[Order]:
        pending = pending or set()
        killed = self.update_kill_switch(equity)
        flatten = minutes_to_close(self.market, now) <= self.cfg.flatten_minutes_before_close
        max_hold = timedelta(minutes=self.horizon_bars * self.bar_minutes)
        orders = []
        for sym, pos in positions.items():
            if pos.qty <= 0 or sym in pending:
                continue
            px = prices.get(sym, pos.last_price or pos.avg_price)
            r = px / pos.avg_price - 1.0
            reason = None
            if killed:
                reason = "daily_loss_limit"
            elif flatten:
                reason = "flatten_before_close"
            elif r <= -self.cfg.stop_loss_pct:
                reason = "stop_loss"
            elif r >= self.cfg.take_profit_pct:
                reason = "take_profit"
            elif now - pos.entry_time >= max_hold:
                reason = "time_exit"
            if reason:
                orders.append(Order(self.market, sym, Side.SELL, pos.qty, now, reason))
        return orders

    # --- entries ---------------------------------------------------------
    def entry_orders(
        self,
        signals: list[Signal],
        positions: dict[str, Position],
        equity: float,
        cash: float,
        now: datetime,
        pending: set[str] | None = None,
    ) -> list[Order]:
        pending = pending or set()
        if self.update_kill_switch(equity):
            return []
        if minutes_to_close(self.market, now) <= self.cfg.no_new_entries_minutes_before_close:
            return []

        open_count = sum(1 for p in positions.values() if p.qty > 0) + len(pending)
        gross = sum(p.market_value for p in positions.values())
        gross_cap = equity * self.cfg.max_gross_exposure_pct
        orders: list[Order] = []

        for s in sorted(signals, key=lambda x: x.expected_return, reverse=True):
            if open_count >= self.cfg.max_positions:
                self._reject("max_positions")
                break
            if s.symbol in positions or s.symbol in pending:
                continue
            if s.prob_up < self.min_prob_up or s.expected_return <= 0:
                self._reject("weak_signal")
                continue
            notional = min(equity * self.cfg.max_position_pct, gross_cap - gross, cash * 0.98)
            qty = math.floor(notional / s.price) if s.price > 0 else 0
            if qty < 1:
                self._reject("no_capacity")
                continue
            edge_bps = s.expected_return * 1e4 - self.costs.round_trip_bps(qty, s.price)
            if edge_bps < self.cfg.min_edge_bps:
                self._reject("edge_below_costs")
                continue
            orders.append(Order(self.market, s.symbol, Side.BUY, qty, now, f"edge={edge_bps:.1f}bps p={s.prob_up:.2f}"))
            open_count += 1
            gross += qty * s.price
            cash -= qty * s.price
        return orders
