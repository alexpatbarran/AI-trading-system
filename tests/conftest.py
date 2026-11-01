from datetime import date

import pytest

from tradebot.config import CostConfig, MarketConfig, Settings
from tradebot.data.synthetic import generate_bars, trading_days
from tradebot.domain import Market


@pytest.fixture
def settings() -> Settings:
    s = Settings(
        markets={
            Market.US: MarketConfig(
                broker="sim",
                universe=[f"S{i}" for i in range(8)],
                min_price=1.0,
                min_bar_dollar_volume=0.0,
                costs=CostConfig(spread_bps=2.0, slippage_bps=3.0),
            ),
            Market.ASX: MarketConfig(
                broker="sim",
                universe=[f"A{i}" for i in range(8)],
                min_price=1.0,
                costs=CostConfig(commission_bps=8.0, min_commission=6.0, spread_bps=6.0, slippage_bps=5.0),
            ),
        }
    )
    s.strategy.train_days = 10
    s.strategy.min_train_days = 5
    s.strategy.retrain_every_days = 5
    s.scanner.min_relative_volume = 1.0
    return s


@pytest.fixture
def us_bars():
    days = trading_days(Market.US, date(2026, 3, 20), 12)
    return generate_bars(Market.US, [f"S{i}" for i in range(8)], days, 5, seed=3)
