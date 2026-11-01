from tradebot.config import StrategyConfig
from tradebot.strategy.base import Strategy

INTRADAY_STRATEGIES = ["momentum", "ml"]
DAILY_STRATEGIES = ["reversal", "momentum", "ml"]


def make_strategy(cfg: StrategyConfig, seed: int = 0) -> Strategy:
    """Intraday strategies."""
    if cfg.name == "momentum":
        from tradebot.strategy.momentum import MomentumStrategy

        return MomentumStrategy()
    if cfg.name == "ml":
        from tradebot.strategy.ml import MLStrategy

        return MLStrategy(seed=seed)
    raise ValueError(f"Unknown intraday strategy: {cfg.name}")


def make_daily_strategy(name: str, seed: int = 0) -> Strategy:
    """Daily/swing strategies.

    reversal: buy last week's biggest losers (short-term reversal)
    momentum: buy the strongest 12-month performers, skipping the last month (12-1 momentum)
    ml:       gradient boosting on cross-sectionally ranked features
    """
    from tradebot.strategy.rank import RankStrategy

    if name == "reversal":
        return RankStrategy(lambda df: 1.0 - df["cs_ret_5d"], name="reversal")
    if name == "momentum":
        return RankStrategy(lambda df: df["cs_mom_12_1"], name="momentum_12_1")
    if name == "ml":
        from tradebot.daily_features import DAILY_ML_FEATURES
        from tradebot.strategy.ml import MLStrategy

        return MLStrategy(features=DAILY_ML_FEATURES, seed=seed)
    raise ValueError(f"Unknown daily strategy: {name}. Choose from {DAILY_STRATEGIES}")


__all__ = ["Strategy", "make_strategy", "make_daily_strategy", "INTRADAY_STRATEGIES", "DAILY_STRATEGIES"]
