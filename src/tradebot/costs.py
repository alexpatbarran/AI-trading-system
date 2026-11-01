"""Transaction cost model: commission, half-spread and slippage.

Spreads can be tiered by liquidity (average daily traded value), because a
thinly traded small cap costs far more to trade than Apple. Tiers are rough
ASSUMPTIONS - measure your real fills once paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.config import CostConfig
from tradebot.domain import Side


@dataclass
class CostModel:
    cfg: CostConfig

    def spread_for(self, avg_dollar_volume: float | None) -> float:
        """Full quoted spread in bps for a stock with this average daily traded value."""
        tiers = self.cfg.spread_tiers
        if not tiers or avg_dollar_volume is None or avg_dollar_volume != avg_dollar_volume:
            return self.cfg.spread_bps
        for min_adv, spread in sorted(tiers, key=lambda t: t[0], reverse=True):
            if avg_dollar_volume >= min_adv:
                return float(spread)
        return float(sorted(tiers, key=lambda t: t[0])[0][1])

    def per_side(self, spread_bps: float | None = None) -> float:
        spread = self.cfg.spread_bps if spread_bps is None else spread_bps
        return spread / 2 + self.cfg.slippage_bps

    @property
    def per_side_bps(self) -> float:
        return self.per_side()

    def fill_price(self, side: Side, ref_price: float, spread_bps: float | None = None) -> float:
        adj = self.per_side(spread_bps) / 1e4
        return ref_price * (1 + adj) if side == Side.BUY else ref_price * (1 - adj)

    def commission(self, qty: int, price: float) -> float:
        if qty <= 0:
            return 0.0
        c = qty * self.cfg.commission_per_share + qty * price * self.cfg.commission_bps / 1e4
        return max(c, self.cfg.min_commission)

    def round_trip_bps(self, qty: int, price: float, spread_bps: float | None = None) -> float:
        """Estimated total cost of entering and exiting, in bps of notional."""
        notional = qty * price
        if notional <= 0:
            return float("inf")
        return 2 * self.per_side(spread_bps) + 2 * self.commission(qty, price) / notional * 1e4
