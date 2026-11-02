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
        cols = ["timestamp", "symbol", "session_date", "close", "fwd_ret"]
        cols += [c for c in ("avg_dollar_volume",) if c in today.columns]
        out.append(today[cols].join(wf.strategy.predict(today)))
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
    # Everything is CROSS-SECTIONAL: compare stocks at the same moment, because that's
    # the choice the strategy makes. Pooling across time would mostly sort predictions
    # by which period they came from (bull vs bear), not by which stock they picked.
    ic_by_time = preds.groupby("timestamp").apply(
        lambda g: _rank_corr(g["expected_return"], g["fwd_ret"]), include_groups=False
    )
    session_of = preds.groupby("timestamp")["session_date"].first()
    daily_ic = ic_by_time.groupby(session_of).mean().dropna()
    pct = preds.groupby("timestamp")["expected_return"].rank(method="first", pct=True)
    preds = preds.assign(decile=np.ceil(pct * 10).clip(1, 10).astype(int))
    deciles = preds.groupby("decile").agg(
        n=("fwd_ret", "size"),
        predicted_bps=("expected_return", lambda x: x.mean() * 1e4),
        realised_bps=("fwd_ret", lambda x: x.mean() * 1e4),
        hit_rate=("fwd_ret", lambda x: (x > 0).mean()),
    )
    if "cost_bps" in preds.columns:  # per-stock costs: what it costs to trade the model's favourites
        round_trip_bps = float(preds.loc[preds["decile"] == 10, "cost_bps"].mean())
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


def diagnose_daily(
    settings: Settings, market: Market, strategy: Strategy, bars: pd.DataFrame, news: pd.DataFrame | None = None
) -> Diagnosis | None:
    from tradebot.daily_backtest import daily_feature_frame, make_walk_forward, required_features, tradeable

    feats = tradeable(daily_feature_frame(settings, market, bars, news), settings, market)
    wf = make_walk_forward(settings, feats, strategy, required_features(news))
    preds = predict_walk_forward(wf)
    if preds.empty:
        return None
    h = settings.daily.horizon_days
    rt = _round_trip(settings, market, float(preds["close"].median()), settings.daily.max_position_pct)
    costs = CostModel(settings.markets[market].costs)
    if costs.cfg.spread_tiers:
        rt_median_price = rt - 2 * costs.per_side()  # commission part for a typical position
        spreads = preds["avg_dollar_volume"].map(costs.spread_for)
        preds = preds.assign(cost_bps=rt_median_price + 2 * (spreads / 2 + costs.cfg.slippage_bps))
    return summarise(preds, market, strategy.name, f"{h} days", h - 1, rt)


def feature_scan(
    settings: Settings, market: Market, bars: pd.DataFrame, news: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Cross-sectional IC of each raw feature against the forward return, over the whole sample.

    Exploratory: there's no training, so no look-ahead, but scanning many features
    and picking the best is itself a form of overfitting. Treat |t| < 3 with suspicion.
    """
    from tradebot.daily_backtest import daily_feature_frame, required_features, tradeable

    feats = tradeable(daily_feature_frame(settings, market, bars, news), settings, market)
    from tradebot.daily_features import MARKET_FEATURES

    required = required_features(news)
    feats = feats.dropna(subset=required + ["fwd_ret"])
    cols = [c for c in required if c not in MARKET_FEATURES]  # same value for every stock on a day
    rows = []
    lags = settings.daily.horizon_days - 1
    for c in cols:
        ic = feats.groupby("timestamp").apply(lambda g, c=c: _rank_corr(g[c], g["fwd_ret"]), include_groups=False)
        ic = ic.dropna()
        rows.append({"feature": c, "ic": float(ic.mean()), "t_stat": newey_west_tstat(ic, lags), "days": len(ic)})
    return pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False).reset_index(drop=True)


FEATURE_NOTES = {
    "ret_1d": "yesterday's return", "ret_5d": "1-week return", "ret_21d": "1-month return",
    "mom_12_1": "12-month momentum", "vol_21d": "volatility", "vol_ratio": "volume vs normal",
    "dist_ma50": "vs 50-day average", "dist_ma200": "vs 200-day average", "dist_52w_high": "vs 52-week high",
    "gap": "overnight gap", "range_pct": "daily high-low range", "rsi_14": "RSI",
    "excess_ret_5d": "1-week return vs market", "excess_ret_21d": "1-month return vs market",
    "market_ret_5d": "market 1-week return", "market_ret_21d": "market 1-month return",
    "market_vol_21d": "market volatility", "news_count_1d": "articles today", "news_abn_5d": "unusual news volume",
    "sent_1d": "headline sentiment today", "sent_5d": "headline sentiment 5d", "event_1d": "news events today",
    "event_5d": "news events 5d",
}


def format_feature_scan(df: pd.DataFrame) -> str:
    lines = [
        "  feature           ic        t-stat  meaning",
        "  (+ = higher value -> higher future return; |t| >= 3 marked strong)",
    ]
    for r in df.itertuples():
        flag = "  <-- strong" if abs(r.t_stat) >= 3 else ""
        lines.append(f"  {r.feature:16s} {r.ic:+.4f}   {r.t_stat:+6.2f}   {FEATURE_NOTES.get(r.feature, '')}{flag}")
    return "\n".join(lines)


def format_diagnosis(d: Diagnosis) -> str:
    lines = [
        f"\n=== {d.market.value} | strategy={d.strategy} | horizon={d.horizon} | signal diagnostics ===",
        f"  out-of-sample days       {d.days}",
        f"  predictions scored       {d.predictions}",
        f"  mean cross-sectional IC  {d.ic_mean:+.4f}",
        f"  IC t-stat                {d.ic_tstat:+.2f}   (|t| < 2: indistinguishable from noise)",
        f"  round-trip cost          {d.round_trip_bps:.1f} bps  (for the model's top-decile picks)",
        f"  average stock            {d.mean_bps:.1f} bps  (what random picks earned over the horizon)",
        "",
        "  decile = rank of the prediction among stocks at the same time (10 = model's favourites)",
        "  decile  predicted_bps  realised_bps  hit_rate        n",
    ]
    for dec, r in d.deciles.iterrows():
        lines.append(f"  {dec:>6}  {r.predicted_bps:>13.1f}  {r.realised_bps:>12.1f}  {r.hit_rate:>8.1%}  {int(r.n):>7}")
    # A long-only pick adds value only if it beats a random pick (the average stock) by more than costs
    excess = d.top_decile_bps - d.mean_bps
    gap = excess - d.round_trip_bps
    if not np.isfinite(d.ic_tstat) or abs(d.ic_tstat) < 2:
        verdict = "No reliable signal. Change features, horizon or universe before tuning anything else."
    elif d.ic_mean <= 0:
        verdict = "Signal is reliably WRONG (negative IC). Check for a bug, or flip the rule."
    elif gap < 0:
        verdict = (
            f"Real signal, but the top decile beats the average stock by only {excess:.1f} bps, "
            f"less than the {d.round_trip_bps:.1f} bps cost."
        )
    else:
        verdict = (
            f"Top decile beats the average stock by {excess:.1f} bps, {gap:.1f} bps more than costs. "
            "Promising: backtest it and check the result holds across years."
        )
    lines += ["", f"  verdict: {verdict}"]
    return "\n".join(lines)
