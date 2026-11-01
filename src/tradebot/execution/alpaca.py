"""Alpaca broker for US equities. Paper endpoint unless live is explicitly allowed."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from tradebot.domain import Market, Order, Position, Side
from tradebot.execution.base import Broker


class AlpacaBroker(Broker):
    market = Market.US

    def __init__(self, live: bool = False, api_key: str | None = None, secret_key: str | None = None):
        from alpaca.trading.client import TradingClient

        self.live = live
        self.client = TradingClient(
            api_key or os.environ["ALPACA_API_KEY"],
            secret_key or os.environ["ALPACA_SECRET_KEY"],
            paper=not live,
        )

    def submit(self, order: Order) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.id,
        )
        self.client.submit_order(req)

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.client.get_all_positions():
            qty = int(float(p.qty))
            if qty == 0:
                continue
            out[p.symbol] = Position(
                Market.US,
                p.symbol,
                qty,
                float(p.avg_entry_price),
                entry_time=datetime.now(timezone.utc),  # engine overrides from its own records
                last_price=float(p.current_price or 0),
            )
        return out

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def cash(self) -> float:
        return float(self.client.get_account().cash)

    def open_order_symbols(self) -> set[str]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {o.symbol for o in orders}

    def cancel_all(self) -> None:
        self.client.cancel_orders()
