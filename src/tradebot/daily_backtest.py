"""Walk-forward backtest for daily/swing trading.

Each day:
  1. orders decided at yesterday's close fill at today's OPEN (with costs);
  2. positions are marked at today's CLOSE;
  3. after the close, the model ranks every liquid stock and decides:
       - stop-loss exits (close below entry by stop_loss_pct);
       - time exits once a position has been held `horizon_days`,
         UNLESS it's still one of today's picks (avoids pointless round trips);
       - new entries for the best-ranked names whose expected return beats
         round-trip costs by min_edge_bps.

Overnight gaps are real here: a stop can fill well below the stop level.
"""

from __future__ import annotations

import math

import pandas as pd

from tradebot.backtest import BacktestResult
from tradebot.config import Settings
from tradebot.costs import CostModel
from tradebot.daily_features import DAILY_FEATURES, add_daily_features, add_daily_labels
from tradebot.domain import Market, Order, Side
from tradebot.execution.sim import SimBroker
from tradebot.markets import SESSIONS, session_bounds
from tradebot.metrics import buy_and_hold_return, compute_metrics
from tradebot.strategy.base import Strategy
from tradebot.walkforward import WalkForward


def daily_feature_frame(
    settings: Settings, market: Market, bars: pd.DataFrame, news: pd.DataFrame | None = None
) -> pd.DataFrame:
    mcfg = settings.markets[market]
    trade_bars = bars[bars["symbol"] != mcfg.benchmark] if mcfg.benchmark else bars
    feats = add_daily_features(trade_bars, market)
    if news is not None:
        from tradebot.news import add_news_features

        feats = add_news_features(feats, news, market)
    return add_daily_labels(feats, settings.daily.horizon_days)


def required_features(news: pd.DataFrame | None) -> list[str]:
    if news is None:
        return list(DAILY_FEATURES)
    from tradebot.news import NEWS_FEATURES

    return DAILY_FEATURES + NEWS_FEATURES


def tradeable(df: pd.DataFrame, settings: Settings, market: Market) -> pd.DataFrame:
    """Liquidity, price and (optional) size-band filter, all point-in-time."""
    d = settings.daily
    ok = (df["close"] >= max(d.min_price, settings.markets[market].min_price)) & (
        df["avg_dollar_volume"] >= d.min_dollar_volume
    )
    if d.size_min_rank is not None:
        ok &= df["size_rank"] >= d.size_min_rank
    if d.size_max_rank is not None:
        ok &= df["size_rank"] <= d.size_max_rank
    return df[ok]


def make_walk_forward(
    settings: Settings, feats: pd.DataFrame, strategy: Strategy, required: list[str] | None = None
) -> WalkForward:
    """Train only on stocks the strategy could actually trade (same size band and liquidity)."""
    d = settings.daily
    return WalkForward(
        feats, strategy, required or DAILY_FEATURES,
        d.train_days, d.min_train_days, d.retrain_every_days, d.min_train_samples,
    )


class DailyBacktester:
    def __init__(self, settings: Settings, market: Market, strategy: Strategy, news: pd.DataFrame | None = None):
        self.s = settings
        self.market = market
        self.strategy = strategy
        self.news = news
        self.required = required_features(news)
        self.mcfg = settings.markets[market]

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        d, mcfg, market = self.s.daily, self.mcfg, self.market
        costs = CostModel(mcfg.costs)
        broker = SimBroker(market, mcfg.starting_capital, costs)
        tz = SESSIONS[market].tz

        feats = daily_feature_frame(self.s, market, bars, self.news)
        wf = make_walk_forward(self.s, tradeable(feats, self.s, market), self.strategy, self.required)
        wf.days = sorted(feats["session_date"].unique())  # step through every day, even ones with no candidates
        by_day = {day: g.set_index("symbol") for day, g in feats.groupby("session_date")}

        rejections: dict[str, int] = {}
        entry_idx: dict[str, int] = {}
        equity_points: list[tuple] = []
        first_day = None
        days_traded = 0

        def reject(reason: str, n: int = 1) -> None:
            rejections[reason] = rejections.get(reason, 0) + n

        for di, day in enumerate(wf.days):
            snap = by_day[day]
            ts_open, ts_close = session_bounds(market, day)

            before = set(broker.positions())
            broker.process_open(snap["open"].dropna().to_dict(), ts_open)
            after = set(broker.positions())
            for sym in after - before:
                entry_idx[sym] = di
            for sym in before - after:
                entry_idx.pop(sym, None)
            closes = snap["close"].to_dict()
            broker.mark(closes)
            broker.cancel_all()  # unfilled orders (e.g. no bar today) expire

            if not wf.step(di):
                equity_points.append((ts_close, broker.equity()))
                continue
            first_day = first_day or day
            days_traded += 1

            # ---- rank today's universe ----
            cands = tradeable(snap.reset_index().dropna(subset=self.required), self.s, market)
            # (symbol, price, expected_return, prob_up, spread_bps)
            picks: list[tuple[str, float, float, float, float]] = []
            if not cands.empty:
                preds = self.strategy.predict(cands)
                ranked = cands.assign(er=preds["expected_return"], p=preds["prob_up"]).sort_values("er", ascending=False)
                equity = broker.equity()
                for r in ranked.itertuples():
                    if len(picks) >= d.max_positions:
                        break
                    if r.p < d.min_prob_up or r.er <= 0:
                        reject("weak_signal")
                        continue
                    qty = math.floor(equity * d.max_position_pct / r.close)
                    if qty < 1:
                        reject("no_capacity")
                        continue
                    spread = costs.spread_for(r.avg_dollar_volume)
                    if r.er * 1e4 - costs.round_trip_bps(qty, r.close, spread) < d.min_edge_bps:
                        reject("edge_below_costs")
                        continue
                    picks.append((r.symbol, r.close, r.er, r.p, spread))
            pick_set = {p[0] for p in picks}
            adv = snap["avg_dollar_volume"].to_dict()

            # ---- exits ----
            positions = broker.positions()
            exiting = set()
            for sym, pos in positions.items():
                px = closes.get(sym, pos.last_price)
                held = di - entry_idx.get(sym, di)
                reason = None
                if px / pos.avg_price - 1 <= -d.stop_loss_pct:
                    reason = "stop_loss"
                elif d.take_profit_pct and px / pos.avg_price - 1 >= d.take_profit_pct:
                    reason = "take_profit"
                elif held >= d.horizon_days and sym not in pick_set:
                    reason = "time_exit"
                elif held >= d.horizon_days:
                    entry_idx[sym] = di  # still a top pick: keep holding, restart the clock
                if reason:
                    spread = costs.spread_for(adv.get(sym))
                    broker.submit(Order(market, sym, Side.SELL, pos.qty, ts_close, reason, spread_bps=spread))
                    exiting.add(sym)

            # ---- entries (sells are queued first, so their proceeds fund buys at the open) ----
            equity = broker.equity()
            held_after = [s for s in positions if s not in exiting]
            gross = sum(positions[s].market_value for s in held_after)
            cash = broker.cash() + sum(positions[s].market_value for s in exiting)
            slots = d.max_positions - len(held_after)
            for sym, price, er, p, spread in picks:
                if sym in positions:
                    continue
                if slots <= 0:
                    reject("max_positions")
                    break
                notional = min(equity * d.max_position_pct, equity * d.max_gross_exposure_pct - gross, cash * 0.98)
                qty = math.floor(notional / price)
                if qty < 1:
                    reject("no_capacity")
                    continue
                reason = f"er={er * 1e4:.0f}bps p={p:.2f}"
                broker.submit(Order(market, sym, Side.BUY, qty, ts_close, reason, spread_bps=spread))
                slots -= 1
                gross += qty * price
                cash -= qty * price

            equity_points.append((ts_close, broker.equity()))

        equity = pd.Series(
            [e for _, e in equity_points], index=pd.DatetimeIndex([t for t, _ in equity_points]), name="equity"
        )
        trades = pd.DataFrame(broker.trades)
        scored_bars = bars
        if first_day is not None:
            start_ts = pd.Timestamp(first_day, tz=tz)
            equity = equity[equity.index >= start_ts]
            scored_bars = bars[bars["timestamp"].dt.tz_convert(tz) >= start_ts]
        trade_bars = scored_bars[scored_bars["symbol"] != mcfg.benchmark] if mcfg.benchmark else scored_bars
        metrics = compute_metrics(
            equity, trades, mcfg.starting_capital, tz, broker.total_commission, broker.total_spread_slippage
        )
        if metrics:
            metrics["trades_per_day"] = metrics["trades"] / max(days_traded, 1)
            metrics["avg_holding_days"] = (
                float((trades["exit_time"] - trades["entry_time"]).dt.days.mean()) if len(trades) else 0.0
            )
        return BacktestResult(
            market=market,
            strategy=f"{self.strategy.name} (daily, {d.horizon_days}d)",
            equity=equity,
            trades=trades,
            metrics=metrics,
            universe_buy_hold=buy_and_hold_return(trade_bars),
            benchmark_buy_hold=buy_and_hold_return(scored_bars, mcfg.benchmark) if mcfg.benchmark else None,
            risk_rejections=rejections,
            days_traded=days_traded,
        )
