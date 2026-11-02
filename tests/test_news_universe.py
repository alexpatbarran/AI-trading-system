from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradebot.config import CostConfig
from tradebot.costs import CostModel
from tradebot.daily_features import add_daily_features
from tradebot.data.synthetic import generate_daily_bars, trading_days
from tradebot.domain import Market
from tradebot.news import NEWS_FEATURES, NewsStore, add_news_features, align_to_sessions, score_headline
from tradebot.research import diagnose_daily
from tradebot.strategy import make_daily_strategy
from tradebot.universe import candidate_assets, select_liquid

UTC = timezone.utc


# ------------------------------------------------------------------ headline scoring


def test_score_headline_events_and_sentiment():
    s, e = score_headline("Acme beats Q3 earnings estimates, raises full-year guidance")
    assert e == 2 and s > 0
    s, e = score_headline("Analyst downgrades Acme to underweight, lowers price target")
    assert e == -2 and s < 0
    assert score_headline("Acme to present at investor conference") == (0.0, 0.0)


# ------------------------------------------------------------------ timing


def test_articles_after_the_close_belong_to_the_next_trading_day():
    days = [date(2026, 7, 15), date(2026, 7, 16), date(2026, 7, 17), date(2026, 7, 20)]
    news = pd.DataFrame({"created_at": pd.to_datetime([
        "2026-07-15 19:59:00",  # 15:59 New York, Wednesday -> Wednesday
        "2026-07-15 20:30:00",  # 16:30 New York, after the close -> Thursday
        "2026-07-18 15:00:00",  # Saturday -> Monday
    ], utc=True)})
    assert list(align_to_sessions(news, Market.US, days)) == [days[0], days[1], days[3]]


def _news(rows):
    return pd.DataFrame(rows, columns=["id", "symbol", "created_at", "headline", "source", "n_symbols"]).assign(
        created_at=lambda d: pd.to_datetime(d["created_at"], utc=True)
    )


def test_news_features_ignore_articles_published_after_the_decision():
    days = trading_days(Market.US, date(2026, 6, 30), 120)
    feats = add_daily_features(generate_daily_bars(Market.US, ["AAA", "BBB"], days, seed=1), Market.US)
    base = [(1, "AAA", "2026-01-02 15:00", "AAA beats earnings estimates", "x", 1)]
    d = days[100]
    after_close = pd.Timestamp(datetime.combine(d, datetime.min.time()), tz="America/New_York") + pd.Timedelta(hours=17)
    late = base + [(2, "AAA", after_close.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"), "AAA beats estimates", "x", 1)]
    a = add_news_features(feats, _news(base), Market.US, warmup_days=10)
    b = add_news_features(feats, _news(late), Market.US, warmup_days=10)
    row = lambda df: df[(df["symbol"] == "AAA") & (df["session_date"] == d)][NEWS_FEATURES]  # noqa: E731
    pd.testing.assert_frame_equal(row(a).reset_index(drop=True), row(b).reset_index(drop=True))
    nxt = days[101]
    assert b[(b["symbol"] == "AAA") & (b["session_date"] == nxt)]["event_1d"].iloc[0] > 0


def test_news_features_are_nan_before_coverage():
    days = trading_days(Market.US, date(2026, 6, 30), 200)
    feats = add_daily_features(generate_daily_bars(Market.US, ["AAA"], days, seed=1), Market.US)
    first = pd.Timestamp(datetime.combine(days[50], datetime.min.time())).tz_localize("America/New_York")
    out = add_news_features(feats, _news([(1, "AAA", (first + pd.Timedelta(hours=12)).tz_convert("UTC"), "x", "x", 1)]),
                            Market.US, warmup_days=20)
    assert out[out["session_date"] < days[70]][NEWS_FEATURES].isna().all().all()
    assert out[out["session_date"] >= days[70]][NEWS_FEATURES].notna().all().all()


def test_stray_old_article_does_not_start_coverage():
    """One 2012-style stray article must not make years of 'no news' look like real data."""
    days = trading_days(Market.US, date(2026, 6, 30), 400)
    feats = add_daily_features(generate_daily_bars(Market.US, ["AAA", "BBB"], days, seed=1), Market.US)
    to_utc = lambda d: (pd.Timestamp(datetime.combine(d, datetime.min.time()), tz="America/New_York")  # noqa: E731
                        + pd.Timedelta(hours=12)).tz_convert("UTC")
    rows = [(0, "AAA", to_utc(days[5]), "stray old article", "x", 1)]
    rows += [(i + 1, "AAA" if i % 2 else "BBB", to_utc(d), "headline", "x", 1) for i, d in enumerate(days[300:])]
    out = add_news_features(feats, _news(rows), Market.US, warmup_days=20)
    assert out[out["session_date"] < days[300]][NEWS_FEATURES].isna().all().all()
    assert out[out["session_date"] >= days[320]][NEWS_FEATURES].notna().all().all()


def test_news_store_roundtrip(tmp_path):
    store = NewsStore(tmp_path / "n.sqlite")
    df = _news([(1, "AAA", "2026-01-02 15:00", "h", "benzinga", 2), (1, "BBB", "2026-01-02 15:00", "h", "benzinga", 2)])
    store.save(df)
    store.save(df)  # duplicates ignored
    assert len(store.load()) == 2
    assert len(store.load(["AAA"])) == 1


def test_news_strategy_finds_planted_news_drift(settings):
    """Stocks get +2% after positive-earnings headlines; the news strategy should detect it."""
    d = settings.daily
    d.train_days, d.min_train_days, d.retrain_every_days, d.min_train_samples = 120, 60, 21, 300
    d.min_dollar_volume, d.min_price = 0, 0
    settings.markets[Market.US].benchmark = None
    syms = [f"N{i}" for i in range(30)]
    days = trading_days(Market.US, date(2026, 6, 30), 500)
    bars = generate_daily_bars(Market.US, syms, days, seed=4)
    bars["session_date"] = bars["timestamp"].dt.tz_convert("America/New_York").dt.date
    rng = np.random.default_rng(0)
    rows, k = [], 0
    for di in range(200, len(days) - 6):
        for sym in rng.choice(syms, 2, replace=False):
            k += 1
            ts = pd.Timestamp(datetime.combine(days[di], datetime.min.time()), tz="America/New_York") + pd.Timedelta(hours=12)
            rows.append((k, sym, ts.tz_convert("UTC"), f"{sym} beats earnings estimates", "x", 1))
            later = (bars["symbol"] == sym) & (bars["session_date"] >= days[di + 2])
            bars.loc[later, ["open", "high", "low", "close"]] *= 1.02
    res = diagnose_daily(settings, Market.US, make_daily_strategy("news", news=True),
                         bars.drop(columns="session_date"), _news(rows))
    assert res is not None and res.ic_tstat > 3


# ------------------------------------------------------------------ universe + costs


def test_candidate_assets_filters_non_common_stock():
    assets = [
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "exchange": "NASDAQ", "asset_class": "us_equity"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA", "asset_class": "us_equity"},
        {"symbol": "ABCDW", "name": "ABCD Corp Warrants", "exchange": "NASDAQ", "asset_class": "us_equity"},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway Class B", "exchange": "NYSE", "asset_class": "us_equity"},
        {"symbol": "XYZ", "name": "XYZ Acquisition Corp", "exchange": "NYSE", "asset_class": "us_equity"},
        {"symbol": "OLDCO", "name": "Old Company Inc", "exchange": "NYSE", "asset_class": "us_equity"},
    ]
    assert candidate_assets(assets) == ["AAPL", "OLDCO"]


def test_select_liquid_keeps_delisted_stocks_that_were_once_liquid():
    months = pd.date_range("2020-01-31", periods=12, freq="ME", tz="UTC")
    rows = []
    for i, ts in enumerate(months):
        rows.append((ts, "BIG", 100.0, 1e7))  # $1bn
        rows.append((ts, "MID", 10.0, 1e7))  # $100m
        rows.append((ts, "TINY", 10.0, 1e3))
        if i < 4:  # very liquid for 4 months, then delisted
            rows.append((ts, "GONE", 50.0, 1e7))  # $500m
        rows.append((ts, "PENNY", 1.0, 1e10))  # liquid but below min price
    monthly = pd.DataFrame(rows, columns=["timestamp", "symbol", "close", "volume"])
    assert select_liquid(monthly, top_n=2, min_months=3, min_price=5) == ["BIG", "GONE", "MID"]


def test_spread_tiers():
    c = CostModel(CostConfig(spread_bps=2, slippage_bps=3, spread_tiers=[(5e8, 2), (2e7, 10), (0, 40)]))
    assert c.spread_for(1e9) == 2
    assert c.spread_for(5e7) == 10
    assert c.spread_for(1e6) == 40
    assert c.round_trip_bps(10, 100, c.spread_for(1e6)) == pytest.approx(2 * (20 + 3))
