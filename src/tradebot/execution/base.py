"""Broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tradebot.domain import Market, Order, Position


class Broker(ABC):
    market: Market

    @abstractmethod
    def submit(self, order: Order) -> None: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    def open_order_symbols(self) -> set[str]:
        return set()

    def cancel_all(self) -> None:  # pragma: no cover - broker specific
        pass
