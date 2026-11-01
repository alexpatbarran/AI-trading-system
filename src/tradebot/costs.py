"""Transaction cost model: commission, half-spread and slippage."""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.config import CostConfig
from tradebot.domain import Side


@dataclass
class CostModel:
    cfg: CostConfig

    @property
    def per_side_bps(self) -> float:
        return self.cfg.spread_bps / 2 + self.cfg.slippage_bps

    def fill_price(self, side: Side, ref_price: float) -> float:
        adj = self.per_side_bps / 1e4
        return ref_price * (1 + adj) if side == Side.BUY else ref_price * (1 - adj)

    def commission(self, qty: int, price: float) -> float:
        if qty <= 0:
            return 0.0
        c = qty * self.cfg.commission_per_share + qty * price * self.cfg.commission_bps / 1e4
        return max(c, self.cfg.min_commission)

    def round_trip_bps(self, qty: int, price: float) -> float:
        """Estimated total cost of entering and exiting, in bps of notional."""
        notional = qty * price
        if notional <= 0:
            return float("inf")
        return 2 * self.per_side_bps + 2 * self.commission(qty, price) / notional * 1e4
