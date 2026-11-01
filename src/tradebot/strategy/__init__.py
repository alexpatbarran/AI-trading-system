from tradebot.config import StrategyConfig
from tradebot.strategy.base import Strategy


def make_strategy(cfg: StrategyConfig, seed: int = 0) -> Strategy:
    if cfg.name == "momentum":
        from tradebot.strategy.momentum import MomentumStrategy

        return MomentumStrategy()
    if cfg.name == "ml":
        from tradebot.strategy.ml import MLStrategy

        return MLStrategy(seed=seed)
    raise ValueError(f"Unknown strategy: {cfg.name}")


__all__ = ["Strategy", "make_strategy"]
