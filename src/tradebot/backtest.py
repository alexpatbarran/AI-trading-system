"""Walk-forward, bar-by-bar backtester.

Guarantees:
  * the strategy is trained only on days strictly before the day it trades;
  * decisions at bar t's close fill at bar t+1's open, with costs;
  * the same scanner, risk manager and cost model run here as in live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from tradebot.config import Settings
from tradebot.costs import CostModel
from tradebot.domain import Market, Signal
from tradebot.execution.sim import SimBroker
from tradebot.features import FEATURES, add_features, add_labels
from tradebot.markets import SESSIONS, regular_session_only
from tradebot.metrics import buy_and_hold_return, compute_metrics
from tradebot.risk import RiskManager
from tradebot.scanner import scan
from tradebot.strategy.base import Strategy


@dataclass
class BacktestResult:
    market: Market
    strategy: str
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    universe_buy_hold: float | None
    benchmark_buy_hold: float | None
    risk_rejections: dict = field(default_factory=dict)
    days_traded: int = 0


class Backtester:
    def __init__(self, settings: Settings, market: Market, strategy: Strategy):
        self.s = settings
        self.market = market
        self.strategy = strategy
        self.mcfg = settings.markets[market]

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        s, mcfg = self.s, self.mcfg
        bar = timedelta(minutes=s.bar_minutes)
        costs = CostModel(mcfg.costs)
        broker = SimBroker(self.market, mcfg.starting_capital, costs)
        risk = RiskManager(
            s.risk, self.market, costs, s.strategy.horizon_bars, s.bar_minutes, s.strategy.min_prob_up
        )

        bars = regular_session_only(bars, self.market)
        benchmark = mcfg.benchmark
        trade_bars = bars[bars["symbol"] != benchmark] if benchmark else bars
        feats = add_labels(add_features(trade_bars, self.market), s.strategy.horizon_bars)
        days = sorted(feats["session_date"].unique())
        by_day = {d: g for d, g in feats.groupby("session_date")}

        equity_points: list[tuple] = []
        last_fit = None
        days_traded = 0
        first_trading_day = None

        for di, day in enumerate(days):
            # ---- walk-forward (re)training on PAST days only ----
            if last_fit is None or di - last_fit >= s.strategy.retrain_every_days:
                window = days[max(0, di - s.strategy.train_days) : di]
                if len(window) >= s.strategy.min_train_days:
                    train = feats[feats["session_date"].isin(window)].dropna(subset=FEATURES + ["fwd_ret"])
                    if len(train) >= s.strategy.min_train_samples and self.strategy.fit(train):
                        last_fit = di

            day_df = by_day[day]
            risk.start_day(broker.equity())
            if self.strategy.fitted:
                days_traded += 1
                first_trading_day = first_trading_day or day
            snaps = {ts: g.set_index("symbol") for ts, g in day_df.groupby("timestamp")}
            stamps = sorted(snaps)

            for ti, ts in enumerate(stamps):
                snap = snaps[ts]
                broker.process_open(snap["open"].to_dict(), ts)
                closes = snap["close"].to_dict()
                broker.mark(closes)
                now = ts + bar  # decision time = bar close

                if ti == len(stamps) - 1:  # last bar: flat into the close
                    broker.close_all_at(closes, now, "session_close")
                    equity_points.append((now, broker.equity()))
                    continue

                pending = broker.open_order_symbols()
                for o in risk.exit_orders(broker.positions(), closes, now, broker.equity(), pending):
                    broker.submit(o)

                if self.strategy.fitted:
                    candidates = scan(snap.reset_index(), mcfg, s.scanner)
                    if not candidates.empty:
                        preds = self.strategy.predict(candidates)
                        signals = [
                            Signal(self.market, r.symbol, now, r.close, p.prob_up, p.expected_return)
                            for r, p in zip(candidates.itertuples(), preds.itertuples())
                        ]
                        for o in risk.entry_orders(
                            signals, broker.positions(), broker.equity(), broker.cash(), now,
                            broker.open_order_symbols(),
                        ):
                            broker.submit(o)

                equity_points.append((now, broker.equity()))
            broker.cancel_all()  # day orders expire

        equity = pd.Series(
            [e for _, e in equity_points], index=pd.DatetimeIndex([t for t, _ in equity_points]), name="equity"
        )
        # Score only the out-of-sample period (from the first day the model could trade)
        if first_trading_day is not None:
            start_ts = pd.Timestamp(first_trading_day, tz=SESSIONS[self.market].tz)
            equity = equity[equity.index >= start_ts]
            local_ts = bars["timestamp"].dt.tz_convert(SESSIONS[self.market].tz)
            bars = bars[local_ts >= start_ts]
            trade_bars = trade_bars[trade_bars["timestamp"].dt.tz_convert(SESSIONS[self.market].tz) >= start_ts]
        trades = pd.DataFrame(broker.trades)
        metrics = compute_metrics(
            equity, trades, mcfg.starting_capital, SESSIONS[self.market].tz,
            broker.total_commission, broker.total_spread_slippage,
        )
        return BacktestResult(
            market=self.market,
            strategy=self.strategy.name,
            equity=equity,
            trades=trades,
            metrics=metrics,
            universe_buy_hold=buy_and_hold_return(trade_bars),
            benchmark_buy_hold=buy_and_hold_return(bars, benchmark) if benchmark else None,
            risk_rejections=dict(risk.rejections),
            days_traded=days_traded,
        )
