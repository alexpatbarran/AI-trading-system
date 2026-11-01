"""Builds data providers and brokers for each market from settings."""

from __future__ import annotations

from tradebot.config import Settings, live_trading_confirmed
from tradebot.costs import CostModel
from tradebot.data.base import DataProvider
from tradebot.domain import Market
from tradebot.execution.base import Broker


def build_data(settings: Settings, market: Market) -> DataProvider:
    kind = settings.markets[market].broker
    if kind == "alpaca":
        from tradebot.data.alpaca import AlpacaData

        return AlpacaData(feed=settings.alpaca.feed)
    if kind == "ibkr":
        from tradebot.data.ibkr import IBKRData

        return IBKRData(_ib(settings), currency=_currency(market))
    from tradebot.data.synthetic import SyntheticData

    return SyntheticData(market)


def build_broker(settings: Settings, market: Market) -> Broker:
    live = settings.mode == "live" and live_trading_confirmed()
    mcfg = settings.markets[market]
    if mcfg.broker == "alpaca":
        from tradebot.execution.alpaca import AlpacaBroker

        return AlpacaBroker(live=live)
    if mcfg.broker == "ibkr":
        from tradebot.execution.ibkr import IBKRBroker

        return IBKRBroker(_ib(settings), currency=_currency(market))
    from tradebot.execution.sim import SimBroker

    return SimBroker(market, mcfg.starting_capital, CostModel(mcfg.costs))


def _ib(settings: Settings):
    from tradebot.ibkr_conn import get_ib

    live = settings.mode == "live" and live_trading_confirmed()
    c = settings.ibkr
    return get_ib(c.host, c.port, c.client_id, allow_live=live)


def _currency(market: Market) -> str:
    return {Market.US: "USD", Market.ASX: "AUD"}[market]
