"""Live / paper trading loop. Runs the same pipeline as the backtester, bar by bar."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from tradebot.config import Settings
from tradebot.costs import CostModel
from tradebot.data.base import DataProvider, drop_incomplete_bars
from tradebot.domain import Market, Side, Signal
from tradebot.execution.base import Broker
from tradebot.features import FEATURES, add_features, add_labels
from tradebot.journal import Journal
from tradebot.markets import is_open, local_date, session_bounds
from tradebot.risk import RiskManager
from tradebot.scanner import scan
from tradebot.strategy.base import Strategy

log = logging.getLogger("tradebot.engine")

DATA_DELAY = timedelta(seconds=20)  # give the feed time to publish the finished bar


class MarketRunner:
    def __init__(
        self,
        settings: Settings,
        market: Market,
        data: DataProvider,
        broker: Broker,
        strategy: Strategy,
        journal: Journal,
    ):
        self.s = settings
        self.market = market
        self.mcfg = settings.markets[market]
        self.data = data
        self.broker = broker
        self.strategy = strategy
        self.journal = journal
        self.bar = timedelta(minutes=settings.bar_minutes)
        self.risk = RiskManager(
            settings.risk, market, CostModel(self.mcfg.costs),
            settings.strategy.horizon_bars, settings.bar_minutes, settings.strategy.min_prob_up,
        )
        self.holidays = set(self.mcfg.holidays)
        self.trading_day = None
        self.days_since_fit: int | None = None
        self.last_decision: datetime | None = None
        self.entry_times = journal.entry_times(market.value)

    # --- scheduling -------------------------------------------------------
    def latest_bar_close(self, now: datetime) -> datetime:
        o, _ = session_bounds(self.market, local_date(self.market, now))
        n = int((now - o) / self.bar)
        return o + n * self.bar

    def due(self, now: datetime) -> bool:
        if not is_open(self.market, now, self.holidays):
            return False
        close = self.latest_bar_close(now)
        return now >= close + DATA_DELAY and (self.last_decision is None or close > self.last_decision)

    # --- sizing -----------------------------------------------------------
    def sleeve_equity(self) -> float:
        # Never size off more than the configured sleeve, even if the account is bigger.
        return min(self.broker.equity(), self.mcfg.starting_capital)

    # --- daily prep ---------------------------------------------------------
    def start_day(self, now: datetime) -> None:
        self.trading_day = local_date(self.market, now)
        self.days_since_fit = None if self.days_since_fit is None else self.days_since_fit + 1
        if self.days_since_fit is None or self.days_since_fit >= self.s.strategy.retrain_every_days:
            self.retrain(now)
        self.risk.start_day(self.sleeve_equity())
        log.info("[%s] new day %s, sleeve equity %.2f", self.market.value, self.trading_day, self.risk.day_start_equity)

    def retrain(self, now: datetime) -> None:
        open_today, _ = session_bounds(self.market, self.trading_day)
        lookback = timedelta(days=int(self.s.strategy.train_days * 1.5) + 4)
        hist = self.data.get_bars(self.mcfg.universe, open_today - lookback, open_today, self.s.bar_minutes)
        if hist.empty:
            log.warning("[%s] no history for training", self.market.value)
            return
        feats = add_labels(add_features(hist, self.market), self.s.strategy.horizon_bars)
        days = sorted(feats["session_date"].unique())[-self.s.strategy.train_days :]
        if len(days) < self.s.strategy.min_train_days:
            log.warning("[%s] only %d days of history; not trading yet", self.market.value, len(days))
            return
        train = feats[feats["session_date"].isin(days)].dropna(subset=FEATURES + ["fwd_ret"])
        if len(train) >= self.s.strategy.min_train_samples and self.strategy.fit(train):
            self.days_since_fit = 0
            log.info("[%s] retrained %s on %d rows", self.market.value, self.strategy.name, len(train))
        else:
            log.warning("[%s] not enough training data (%d rows)", self.market.value, len(train))

    # --- one decision ---------------------------------------------------------
    def step(self, now: datetime) -> None:
        if self.trading_day != local_date(self.market, now):
            self.start_day(now)
        decision_close = self.latest_bar_close(now)
        self.last_decision = decision_close

        bars = self.data.get_bars(self.mcfg.universe, now - timedelta(days=4), now, self.s.bar_minutes)
        bars = drop_incomplete_bars(bars, now, self.s.bar_minutes)
        if bars.empty:
            log.warning("[%s] no bars", self.market.value)
            return
        feats = add_features(bars, self.market)
        snap = feats[feats["timestamp"] == feats["timestamp"].max()]
        closes = dict(zip(snap["symbol"], snap["close"]))

        positions = self.broker.positions()
        for sym, pos in positions.items():
            pos.entry_time = self.entry_times.get(sym, pos.entry_time)
            pos.last_price = closes.get(sym, pos.last_price)
        equity = self.sleeve_equity()
        gross = sum(p.market_value for p in positions.values())
        cash = min(self.broker.cash(), equity - gross)
        pending = self.broker.open_order_symbols()

        orders = self.risk.exit_orders(positions, closes, decision_close, equity, pending)
        signals: list[Signal] = []
        if self.strategy.fitted:
            candidates = scan(snap, self.mcfg, self.s.scanner)
            if not candidates.empty:
                preds = self.strategy.predict(candidates)
                signals = [
                    Signal(self.market, r.symbol, decision_close, r.close, p.prob_up, p.expected_return)
                    for r, p in zip(candidates.itertuples(), preds.itertuples())
                ]
                orders += self.risk.entry_orders(signals, positions, equity, cash, decision_close, pending)

        self.journal.signals(signals)
        for o in orders:
            log.info("[%s] %s %d %s (%s)", self.market.value, o.side.value.upper(), o.qty, o.symbol, o.reason)
            self.broker.submit(o)
            self.journal.order(o, self.s.mode)
            entry = decision_close if o.side == Side.BUY else None
            self.journal.set_entry(self.market.value, o.symbol, entry)
            if entry:
                self.entry_times[o.symbol] = entry
            else:
                self.entry_times.pop(o.symbol, None)
        self.journal.equity(decision_close, self.market.value, equity, cash, len(positions))


class Engine:
    def __init__(self, runners: list[MarketRunner], poll_seconds: float = 10):
        self.runners = runners
        self.poll_seconds = poll_seconds

    def run(self) -> None:
        log.info("Engine started for %s", ", ".join(r.market.value for r in self.runners))
        try:
            while True:
                now = datetime.now(timezone.utc)
                for r in self.runners:
                    if r.due(now):
                        try:
                            r.step(now)
                        except Exception:  # keep the other market alive
                            log.exception("[%s] step failed", r.market.value)
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            log.warning("Stopped by user. Open positions are NOT closed automatically:")
            for r in self.runners:
                for sym, p in r.broker.positions().items():
                    log.warning("  [%s] %s qty=%d", r.market.value, sym, p.qty)
