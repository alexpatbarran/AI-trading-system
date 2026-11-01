"""Signal diagnostics: does the model predict anything, and is it enough to beat costs?

A backtest with zero trades can't tell "no signal" apart from "real signal
that's smaller than costs". This module scores walk-forward predictions on
EVERY symbol and bar (ignoring the scanner and risk gates):

  * Information coefficient (IC): daily rank correlation between predicted and
    realised forward return. |t-stat| above ~2-3 suggests real signal.
  * Decile table: realised return by prediction decile. If the top decile's
    average return is below round-trip costs, the edge can't be traded at this
    frequency, however real it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.config import Settings
from tradebot.costs import CostModel
from tradebot.domain import Market
from tradebot.features import FEATURES, add_features, add_labels
from tradebot.markets import regular_session_only
from tradebot.strategy.base import Strategy


@dataclass
class Diagnosis:
    market: Market
    strategy: str
    days: int
    predictions: int
    ic_mean: float
    ic_tstat: float
    deciles: pd.DataFrame
    round_trip_bps: float

    @property
    def top_decile_bps(self) -> float:
        return float(self.deciles["realised_bps"].iloc[-1]) if len(self.deciles) else float("nan")


def walk_forward_predictions(settings: Settings, market: Market, strategy: Strategy, bars: pd.DataFrame) -> pd.DataFrame:
    s = settings.strategy
    mcfg = settings.markets[market]
    bars = regular_session_only(bars, market)
    if mcfg.benchmark:
        bars = bars[bars["symbol"] != mcfg.benchmark]
    feats = add_labels(add_features(bars, market), s.horizon_bars)
    feats = feats.dropna(subset=FEATURES + ["fwd_ret"])
    days = sorted(feats["session_date"].unique())
    out, last_fit = [], None
    for di, day in enumerate(days):
        if last_fit is None or di - last_fit >= s.retrain_every_days:
            window = days[max(0, di - s.train_days) : di]
            if len(window) >= s.min_train_days:
                train = feats[feats["session_date"].isin(window)]
                if len(train) >= s.min_train_samples and strategy.fit(train):
                    last_fit = di
        if not strategy.fitted:
            continue
        today = feats[feats["session_date"] == day]
        pred = strategy.predict(today)
        out.append(today[["timestamp", "symbol", "session_date", "close", "fwd_ret"]].join(pred))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def diagnose(settings: Settings, market: Market, strategy: Strategy, bars: pd.DataFrame) -> Diagnosis | None:
    preds = walk_forward_predictions(settings, market, strategy, bars)
    if preds.empty:
        return None
    daily_ic = preds.groupby("session_date").apply(
        lambda g: _rank_corr(g["expected_return"], g["fwd_ret"]), include_groups=False
    ).dropna()
    ic_mean = float(daily_ic.mean()) if len(daily_ic) else np.nan
    ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else np.nan
    ic_t = ic_mean / ic_std * np.sqrt(len(daily_ic)) if ic_std and ic_std > 0 else np.nan

    ranks = preds["expected_return"].rank(method="first")
    preds["decile"] = pd.qcut(ranks, 10, labels=False) + 1
    deciles = preds.groupby("decile").agg(
        n=("fwd_ret", "size"),
        predicted_bps=("expected_return", lambda x: x.mean() * 1e4),
        realised_bps=("fwd_ret", lambda x: x.mean() * 1e4),
        hit_rate=("fwd_ret", lambda x: (x > 0).mean()),
    )

    mcfg = settings.markets[market]
    price = float(preds["close"].median())
    qty = max(1, int(mcfg.starting_capital * settings.risk.max_position_pct / price))
    rt = CostModel(mcfg.costs).round_trip_bps(qty, price)
    return Diagnosis(market, strategy.name, len(daily_ic), len(preds), ic_mean, float(ic_t), deciles, rt)


def format_diagnosis(d: Diagnosis) -> str:
    lines = [
        f"\n=== {d.market.value} | strategy={d.strategy} | signal diagnostics ===",
        f"  out-of-sample days       {d.days}",
        f"  predictions scored       {d.predictions}",
        f"  mean daily IC            {d.ic_mean:+.4f}",
        f"  IC t-stat                {d.ic_tstat:+.2f}   (|t| < 2: indistinguishable from noise)",
        f"  round-trip cost          {d.round_trip_bps:.1f} bps  (typical position)",
        "",
        "  decile  predicted_bps  realised_bps  hit_rate       n",
    ]
    for dec, r in d.deciles.iterrows():
        lines.append(f"  {dec:>6}  {r.predicted_bps:>13.1f}  {r.realised_bps:>12.1f}  {r.hit_rate:>8.1%}  {int(r.n):>6}")
    gap = d.top_decile_bps - d.round_trip_bps
    if not np.isfinite(d.ic_tstat) or abs(d.ic_tstat) < 2:
        verdict = "No reliable signal. Change features, horizon or universe before tuning anything else."
    elif d.ic_mean <= 0:
        verdict = "Signal is reliably WRONG (negative IC). Check for a bug, or the effect may be mean reversion."
    elif gap < 0:
        verdict = f"Real signal, but the top decile is {abs(gap):.1f} bps short of covering costs. Try a longer horizon or cheaper execution."
    else:
        verdict = f"Top decile beats costs by {gap:.1f} bps. Promising: confirm on more data before paper trading."
    lines += ["", f"  verdict: {verdict}"]
    return "\n".join(lines)
