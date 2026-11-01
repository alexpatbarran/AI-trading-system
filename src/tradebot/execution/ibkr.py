"""Interactive Brokers broker for ASX equities.

Paper vs live is decided by which TWS/Gateway session you log into and the
port (see ibkr_conn.LIVE_PORTS). The connection helper refuses live ports
unless live trading is confirmed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tradebot.domain import Market, Order, Position, Side
from tradebot.execution.base import Broker


class IBKRBroker(Broker):
    market = Market.ASX

    def __init__(self, ib, exchange: str = "ASX", currency: str = "AUD"):
        self.ib = ib
        self.exchange = exchange
        self.currency = currency

    def _contract(self, symbol: str):
        from ib_async import Stock

        c = Stock(symbol, self.exchange, self.currency)
        self.ib.qualifyContracts(c)
        return c

    def submit(self, order: Order) -> None:
        from ib_async import MarketOrder

        action = "BUY" if order.side == Side.BUY else "SELL"
        ib_order = MarketOrder(action, order.qty)
        ib_order.orderRef = order.id
        self.ib.placeOrder(self._contract(order.symbol), ib_order)

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.ib.positions():
            if p.contract.currency != self.currency or p.position == 0:
                continue
            out[p.contract.symbol] = Position(
                Market.ASX,
                p.contract.symbol,
                int(p.position),
                float(p.avgCost),
                entry_time=datetime.now(timezone.utc),  # engine overrides from its own records
            )
        return out

    def _account_value(self, tag: str) -> float:
        values = [v for v in self.ib.accountValues() if v.tag == tag]
        for currency in (self.currency, "BASE"):
            for v in values:
                if v.currency == currency:
                    return float(v.value)
        raise RuntimeError(f"IBKR account value {tag} not available")

    def equity(self) -> float:
        return self._account_value("NetLiquidation")

    def cash(self) -> float:
        return self._account_value("TotalCashValue")

    def open_order_symbols(self) -> set[str]:
        return {t.contract.symbol for t in self.ib.openTrades() if t.contract.currency == self.currency}

    def cancel_all(self) -> None:
        for t in self.ib.openTrades():
            if t.contract.currency == self.currency:
                self.ib.cancelOrder(t.order)
