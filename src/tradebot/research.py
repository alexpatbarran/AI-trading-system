"""Signal diagnostics: does the model predict anything, and is it enough to beat costs?

A backtest with zero trades can't tell "no signal" apart from "real signal
that's smaller than costs". This module scores walk-forward predictions on
EVERY tradeable symbol (ignoring the scanner and position limits):

  * Information coefficient (IC): daily rank correlation between predicted and
    realised forward return. |t-stat| above ~2-3 suggests real signal.
    For multi-day labels, consecutive days' labels overlap, which inflates a
    naive t-stat, so a Newey-West correction is applied.
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
from tradebot.walkforward import WalkForward


@dataclass
class Diagnosis:
    market: Market
    strategy: str
    horizon: str
    days: int
    predictions: int
    ic_mean: float
    ic_tstat: float
    deciles: pd.DataFrame
    round_trip_bps: float

    @property
    def top_decile_bps(self) -> float:
        return float(self.deciles["realised_bps"].iloc[-1]) if len(self.deciles) else float("nan")

    @property
    def mean_bps(self) -> float:
        d = self.deciles
        return float((d["realised_bps"] * d["n"]).sum() / d["n"].sum()) if len(d) else float("nan")

    @property
    def bottom_decile_bps(self) -> float:
        return float(self.deciles["realised_bps"].iloc[0]) if len(self.deciles) else float("nan")


def predict_walk_forward(wf: WalkForward, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for every day the model is trained. `rows` limits what gets scored."""
    rows = wf.feats if rows is None else rows
    rows = rows.dropna(subset=wf.required + ["fwd_ret"])
    by_day = {d: g for d, g in rows.groupby("session_date")}
    out = []
    for di, day in enumerate(wf.days):
        if not wf.step(di) or day not in by_day:
            continue
        today = by_day[day]
        out.append(today[["timestamp", "symbol", "session_date", "close", "fwd_ret"]].join(wf.strategy.predict(today)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    mean = x.mean()
    e = x - mean
    s = (e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(mean / np.sqrt(s / n)) if s > 0 else float("nan")


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def summarise(
    preds: pd.DataFrame, market: Market, strategy: str, horizon: str, overlap_lags: int, round_trip_bps: float
) -> Diagnosis | None:
    if preds.empty:
        return None
    daily_ic = preds.groupby("session_date").apply(
        lambda g: _rank_corr(g["expected_return"], g["fwd_ret"]), include_groups=False
    ).dropna()
    ranks = preds["expected_return"].rank(method="first")
    preds = preds.assign(decile=pd.qcut(ranks, 10, labels=False) + 1)
    deciles = preds.groupby("decile").agg(
        n=("fwd_ret", "size"),
        predicted_bps=("expected_return", lambda x: x.mean() * 1e4),
        realised_bps=("fwd_ret", lambda x: x.mean() * 1e4),
        hit_rate=("fwd_ret", lambda x: (x > 0).mean()),
    )
    return Diagnosis(
        market, strategy, horizon, len(daily_ic), len(preds),
        float(daily_ic.mean()) if len(daily_ic) else np.nan,
        newey_west_tstat(daily_ic, overlap_lags), deciles, round_trip_bps,
    )


def _round_trip(settings: Settings, market: Market, price: float, position_pct: float) -> float:
    mcfg = settings.markets[market]
    qty = max(1, int(mcfg.starting_capital * position_pct / price))
    return CostModel(mcfg.costs).round_trip_bps(qty, price)


def diagnose(settings: Settings, market: Market, strategy: Strategy, bars: pd.DataFrame) -> Diagnosis | None:
    """Intraday diagnostics."""
    s = settings.strategy
    mcfg = settings.markets[market]
    bars = regular_session_only(bars, market)
    if mcfg.benchmark:
        bars = bars[bars["symbol"] != mcfg.benchmark]
    feats = add_labels(add_features(bars, market), s.horizon_bars)
    wf = WalkForward(feats, strategy, FEATURES, s.train_days, s.min_train_days, s.retrain_every_days, s.min_train_samples)
    preds = predict_walk_forward(wf)
    if preds.empty:
        return None
    rt = _round_trip(settings, market, float(preds["close"].median()), settings.risk.max_position_pct)
    horizon = f"{s.horizon_bars * settings.bar_minutes} min"
    return summarise(preds, market, strategy.name, horizon, 0, rt)


def diagnose_daily(settings: Settings, market: Market, strategy: Strategy, bars: pd.DataFrame) -> Diagnosis | None:
    from tradebot.daily_backtest import daily_feature_frame, make_walk_forward, tradeable

    feats = daily_feature_frame(settings, market, bars)
    wf = make_walk_forward(settings, feats, strategy)
    preds = predict_walk_forward(wf, tradeable(feats, settings, market))
    if preds.empty:
        return None
    h = settings.daily.horizon_days
    rt = _round_trip(settings, market, float(preds["close"].median()), settings.daily.max_position_pct)
    return summarise(preds, market, strategy.name, f"{h} days", h - 1, rt)


def format_diagnosis(d: Diagnosis) -> str:
    lines = [
        f"\n=== {d.market.value} | strategy={d.strategy} | horizon={d.horizon} | signal diagnostics ===",
        f"  out-of-sample days       {d.days}",
        f"  predictions scored       {d.predictions}",
        f"  mean daily IC            {d.ic_mean:+.4f}",
        f"  IC t-stat                {d.ic_tstat:+.2f}   (|t| < 2: indistinguishable from noise)",
        f"  round-trip cost          {d.round_trip_bps:.1f} bps  (typical position)",
        f"  average stock            {d.mean_bps:.1f} bps  (what random picks earned over the horizon)",
        "",
        "  decile  predicted_bps  realised_bps  hit_rate        n",
    ]
    for dec, r in d.deciles.iterrows():
        lines.append(f"  {dec:>6}  {r.predicted_bps:>13.1f}  {r.realised_bps:>12.1f}  {r.hit_rate:>8.1%}  {int(r.n):>7}")
    gap = d.top_decile_bps - d.round_trip_bps
    if not np.isfinite(d.ic_tstat) or abs(d.ic_tstat) < 2:
        verdict = "No reliable signal. Change features, horizon or universe before tuning anything else."
    elif d.ic_mean <= 0:
        verdict = "Signal is reliably WRONG (negative IC). Check for a bug, or flip the rule."
    elif gap < 0:
        verdict = f"Real signal, but the top decile is {abs(gap):.1f} bps short of covering costs."
    else:
        spread = d.top_decile_bps - d.bottom_decile_bps
        verdict = (
            f"Top decile beats costs by {gap:.1f} bps (top-bottom spread {spread:.1f} bps). "
            "Check it against simply holding the universe before trusting it."
        )
    lines += ["", f"  verdict: {verdict}"]
    return "\n".join(lines)
