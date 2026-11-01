import numpy as np
import pandas as pd

from tradebot.domain import Market
from tradebot.features import FEATURES, add_features, add_labels


def test_features_have_no_lookahead(us_bars):
    """Features at time t must be identical whether or not future bars exist."""
    full = add_features(us_bars, Market.US)
    stamps = sorted(us_bars["timestamp"].unique())
    for cut in (stamps[100], stamps[400], stamps[-50]):
        truncated = add_features(us_bars[us_bars["timestamp"] <= cut], Market.US)
        a = full[full["timestamp"] == cut].set_index("symbol")[FEATURES].sort_index()
        b = truncated[truncated["timestamp"] == cut].set_index("symbol")[FEATURES].sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_labels_never_cross_sessions(us_bars):
    df = add_labels(add_features(us_bars, Market.US), horizon_bars=6)
    last_bars = df.groupby(["symbol", "session_date"]).tail(6)
    assert last_bars["fwd_ret"].isna().all()


def test_label_matches_next_open_to_future_close(us_bars):
    df = add_labels(add_features(us_bars, Market.US), horizon_bars=3)
    sym = df[df["symbol"] == "S0"].reset_index(drop=True)
    i = 10
    expected = sym.loc[i + 3, "close"] / sym.loc[i + 1, "open"] - 1
    assert np.isclose(sym.loc[i, "fwd_ret"], expected)
