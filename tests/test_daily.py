from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradebot.config import read_universe
from tradebot.daily_backtest import DailyBacktester, daily_feature_frame, make_walk_forward
from tradebot.daily_features import DAILY_FEATURES, add_daily_features
from tradebot.data.synthetic import generate_daily_bars, trading_days
from tradebot.domain import Market
from tradebot.research import diagnose_daily, newey_west_tstat
from tradebot.strategy import make_daily_strategy
from tradebot.strategy.base import Strategy

SYMS = [f"D{i}" for i in range(30)]


@pytest.fixture
def daily_settings(settings):
    d = settings.daily
    d.train_days, d.min_train_days, d.retrain_every_days, d.min_train_samples = 120, 60, 21, 500
    d.min_dollar_volume = 0
    settings.markets[Market.US].universe = SYMS
    settings.markets[Market.US].benchmark = None
    return settings


def _bars(momentum=0.0, n_days=420, seed=1):
    days = trading_days(Market.US, date(2026, 6, 30), n_days)
    return generate_daily_bars(Market.US, SYMS, days, momentum=momentum, seed=seed)


def test_daily_features_have_no_lookahead():
    bars = _bars(n_days=320)
    full = add_daily_features(bars, Market.US)
    dates = sorted(full["session_date"].unique())
    for cut in (dates[260], dates[-10]):
        trunc = add_daily_features(bars[bars["timestamp"].dt.tz_convert("America/New_York").dt.date <= cut], Market.US)
        a = full[full["session_date"] == cut].set_index("symbol")[DAILY_FEATURES].sort_index()
        b = trunc[trunc["session_date"] == cut].set_index("symbol")[DAILY_FEATURES].sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


class SpyStrategy(Strategy):
    """Records what it was trained on."""

    name = "spy"

    def fit(self, train):
        self.last_max_label_end = max(train["label_end_date"])
        self.fitted = True
        return True

    def predict(self, rows):
        return pd.DataFrame({"prob_up": 0.5, "expected_return": 0.0}, index=rows.index)


def test_walk_forward_purges_overlapping_labels(daily_settings):
    feats = daily_feature_frame(daily_settings, Market.US, _bars(n_days=400))
    spy = SpyStrategy()
    wf = make_walk_forward(daily_settings, feats, spy)
    checked = 0
    for di, day in enumerate(wf.days):
        before = wf.fits
        wf.step(di)
        if wf.fits > before:
            assert spy.last_max_label_end < day  # no training label reaches the trading day
            checked += 1
    assert checked >= 2


def test_daily_backtest_finds_planted_reversal_and_holds_overnight(daily_settings):
    r = DailyBacktester(daily_settings, Market.US, make_daily_strategy("reversal")).run(_bars(momentum=-0.4))
    assert r.metrics["trades"] > 20
    assert r.metrics["total_return"] > 0
    held = (r.trades["exit_time"] - r.trades["entry_time"]).dt.days
    assert (held >= 1).all()  # bought at an open, sold at a later open


def test_daily_diagnose_detects_planted_reversal(daily_settings):
    d = diagnose_daily(daily_settings, Market.US, make_daily_strategy("reversal"), _bars(momentum=-0.4))
    assert d.ic_tstat > 3
    assert d.top_decile_bps > d.bottom_decile_bps


def test_newey_west_matches_plain_t_for_iid():
    x = pd.Series(np.random.default_rng(0).normal(0.1, 1, 5000))
    plain = x.mean() / x.std() * np.sqrt(len(x))
    assert newey_west_tstat(x, 0) == pytest.approx(plain, rel=0.01)
    assert newey_west_tstat(x, 4) == pytest.approx(plain, rel=0.1)


def test_read_universe(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("# header\nAAPL MSFT\nNVDA  # comment\n\nAAPL\n")
    assert read_universe(f) == ["AAPL", "MSFT", "NVDA"]
    assert len(read_universe("config/universe_us.txt")) >= 100
