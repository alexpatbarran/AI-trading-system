"""Simulated broker used by the backtester.

Orders submitted at a bar's close fill at the NEXT bar's open, with spread,
slippage and commission applied. Same-bar fills would be look-ahead.
"""

from __future__ import annotations

from datetime import datetime

from tradebot.costs import CostModel
from tradebot.domain import Fill, Market, Order, Position, Side
from tradebot.execution.base import Broker


class SimBroker(Broker):
    def __init__(self, market: Market, starting_cash: float, costs: CostModel):
        self.market = market
        self._cash = starting_cash
        self.costs = costs
        self.pending: list[Order] = []
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.trades: list[dict] = []
        self.total_commission = 0.0
        self.total_spread_slippage = 0.0

    # --- Broker API -------------------------------------------------------
    def submit(self, order: Order) -> None:
        self.pending.append(order)

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    def open_order_symbols(self) -> set[str]:
        return {o.symbol for o in self.pending}

    def cancel_all(self) -> None:
        self.pending.clear()

    # --- simulation hooks -------------------------------------------------
    def process_open(self, opens: dict[str, float], ts: datetime) -> None:
        still_pending = []
        for o in self.pending:
            if o.symbol not in opens:
                still_pending.append(o)
                continue
            self._execute(o, opens[o.symbol], ts)
        self.pending = still_pending

    def mark(self, closes: dict[str, float]) -> None:
        for sym, pos in self._positions.items():
            if sym in closes:
                pos.last_price = closes[sym]

    def close_all_at(self, closes: dict[str, float], ts: datetime, reason: str) -> None:
        self.pending.clear()
        for sym, pos in list(self._positions.items()):
            px = closes.get(sym, pos.last_price or pos.avg_price)
            self._execute(Order(self.market, sym, Side.SELL, pos.qty, ts, reason), px, ts)

    # --- internals ----------------------------------------------------------
    def _execute(self, o: Order, ref_price: float, ts: datetime) -> None:
        price = self.costs.fill_price(o.side, ref_price)
        if o.side == Side.BUY:
            qty = o.qty
            commission = self.costs.commission(qty, price)
            while qty > 0 and qty * price + commission > self._cash:
                qty -= 1
                commission = self.costs.commission(qty, price)
            if qty <= 0:
                return
            slip = (price - ref_price) * qty
            self._cash -= qty * price + commission
            # avg_price already includes entry spread/slippage; entry_costs holds the commission
            self._positions[o.symbol] = Position(
                self.market, o.symbol, qty, price, ts, last_price=ref_price, entry_costs=commission
            )
        else:
            pos = self._positions.get(o.symbol)
            if pos is None:
                return
            qty = min(o.qty, pos.qty)
            commission = self.costs.commission(qty, price)
            slip = (ref_price - price) * qty
            self._cash += qty * price - commission
            entry_commission = pos.entry_costs * qty / pos.qty
            pos.entry_costs -= entry_commission
            pnl = (price - pos.avg_price) * qty - commission - entry_commission
            self.trades.append(
                {
                    "symbol": o.symbol,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "qty": qty,
                    "entry_price": pos.avg_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "return": pnl / (pos.avg_price * qty),
                    "exit_reason": o.reason,
                }
            )
            pos.qty -= qty
            if pos.qty <= 0:
                del self._positions[o.symbol]
        self.total_commission += commission
        self.total_spread_slippage += slip
        self.fills.append(Fill(o.id, self.market, o.symbol, o.side, qty, price, commission, ts, o.reason))
