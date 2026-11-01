"""Settings loading and live-trading safety checks."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from tradebot.domain import Market

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
LIVE_CONFIRM_ENV = "TRADEBOT_CONFIRM_LIVE"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_THE_RISKS"


class CostConfig(BaseModel):
    commission_per_share: float = 0.0
    commission_bps: float = 0.0
    min_commission: float = 0.0
    spread_bps: float = 2.0
    slippage_bps: float = 3.0


class MarketConfig(BaseModel):
    enabled: bool = True
    broker: Literal["alpaca", "ibkr", "sim"] = "sim"
    starting_capital: float = 10_000.0
    benchmark: str | None = None
    min_price: float = 1.0
    min_bar_dollar_volume: float = 0.0
    holidays: list[date] = Field(default_factory=list)
    universe: list[str] = Field(default_factory=list)
    costs: CostConfig = Field(default_factory=CostConfig)


class StrategyConfig(BaseModel):
    name: Literal["momentum", "ml"] = "momentum"
    horizon_bars: int = 6
    train_days: int = 20
    min_train_days: int = 10
    retrain_every_days: int = 5
    min_train_samples: int = 500
    min_prob_up: float = 0.52


class ScannerConfig(BaseModel):
    top_n: int = 10
    min_relative_volume: float = 1.2


class RiskConfig(BaseModel):
    max_position_pct: float = 0.10
    max_positions: int = 5
    max_gross_exposure_pct: float = 0.60
    daily_loss_limit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    min_edge_bps: float = 5.0
    no_new_entries_minutes_before_close: float = 30
    flatten_minutes_before_close: float = 10


class AlpacaConfig(BaseModel):
    feed: Literal["iex", "sip"] = "iex"


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17


class Settings(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    bar_minutes: int = 5
    db_path: str = "data/tradebot.sqlite"
    markets: dict[Market, MarketConfig] = Field(default_factory=dict)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)

    def enabled_markets(self) -> list[Market]:
        return [m for m, c in self.markets.items() if c.enabled]


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_settings(path: str | Path | None = None) -> Settings:
    load_dotenv()
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings.model_validate(raw)


def live_trading_confirmed() -> bool:
    return os.environ.get(LIVE_CONFIRM_ENV) == LIVE_CONFIRM_VALUE


def assert_live_allowed(settings: Settings) -> None:
    """Raise unless live trading has been explicitly enabled twice over."""
    if settings.mode != "live":
        raise PermissionError("Config mode is not 'live'.")
    if not live_trading_confirmed():
        raise PermissionError(
            f"Live trading needs {LIVE_CONFIRM_ENV}={LIVE_CONFIRM_VALUE} in the environment."
        )
