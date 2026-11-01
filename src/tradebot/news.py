"""News features from Alpaca's news API (Benzinga feed, US stocks).

Timing rule (the look-ahead trap with news): an article counts for trading day
d only if it was published at or before day d's CLOSE. Anything published
after the close - including after-hours earnings releases - belongs to the
next trading day. The label starts at the next OPEN, so the overnight gap
reaction is NOT captured: only drift that continues after the open is.

Features (per stock, per trading day):
  * attention: article count, and the 5-day count versus its own 60-day norm
  * headline sentiment: a small finance keyword lexicon (transparent, no black box)
  * events: analyst upgrades/downgrades, price-target changes, earnings beats/misses,
    guidance raises/cuts, scored +1/-1 by regex
Articles tagged with many symbols (market wrap-ups, "stocks moving premarket")
are down-weighted by 1/n_symbols and ignored for sentiment beyond 3 symbols.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.domain import Market
from tradebot.markets import SESSIONS, session_bounds

log = logging.getLogger("tradebot.news")

NEWS_FEATURES = ["news_count_1d", "news_abn_5d", "sent_1d", "sent_5d", "event_1d", "event_5d"]
MAX_SYMBOLS_FOR_SENTIMENT = 3

POSITIVE = set(
    """beat beats tops exceeds surpasses record strong stronger soar soars soared surge surges surged
    jump jumps jumped rally rallies rallied gain gains climbs climb rises rise rose approval approved
    wins win awarded upgrade upgraded upgrades outperform overweight bullish buyback repurchase
    raises raised boosts boost expands expansion growth profitable breakthrough partnership accelerates""".split()
)
NEGATIVE = set(
    """miss misses missed weak weaker plunge plunges plunged tumble tumbles tumbled slump slumps sink sinks
    sank drop drops dropped fall falls fell decline declines declined slides slid lawsuit sued probe
    investigation subpoena recall delay delays delayed halt halts layoffs warns warning bankruptcy fraud
    resigns downgrade downgraded downgrades underperform underweight bearish cuts cut lowers lowered
    loss losses shortfall disappointing concerns fine fined default restatement""".split()
)

_EARN = r"(estimates?|expectations|consensus|eps|earnings|revenue|sales|profit)"
EVENTS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\bupgrad"), +1),
    (re.compile(r"\bdowngrad"), -1),
    (re.compile(r"\b(raises?|raised|boosts?|lifts?|increases?)\b.{0,40}price target|price target (raised|increased|boosted)"), +1),
    (re.compile(r"\b(lowers?|lowered|cuts?|reduces?|trims?)\b.{0,40}price target|price target (lowered|cut|reduced)"), -1),
    (re.compile(rf"\b(beats?|tops?|exceeds?|surpass\w*)\b.{{0,40}}\b{_EARN}\b|\b{_EARN}\b.{{0,40}}\b(beats?|tops|above)\b"), +1),
    (re.compile(rf"\bmiss(es|ed)?\b.{{0,40}}\b{_EARN}\b|\b{_EARN}\b.{{0,40}}\b(miss(es|ed)?|below)\b"), -1),
    (re.compile(r"\b(raises?|raised|boosts?|lifts?|ups)\b.{0,30}\b(guidance|outlook|forecast)"), +1),
    (re.compile(r"\b(lowers?|lowered|cuts?|slashes|reduces?)\b.{0,30}\b(guidance|outlook|forecast)"), -1),
]
_WORD = re.compile(r"[a-z']+")


def score_headline(headline: str) -> tuple[float, float]:
    """(sentiment, event) for one headline. Sentiment in [-3, 3], event in [-2, 2]."""
    h = (headline or "").lower()
    words = _WORD.findall(h)
    sentiment = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    event = sum(sign for pattern, sign in EVENTS if pattern.search(h))
    return float(np.clip(sentiment, -3, 3)), float(np.clip(event, -2, 2))


# --------------------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id INTEGER NOT NULL, symbol TEXT NOT NULL, created_at TEXT NOT NULL,
    headline TEXT, source TEXT, n_symbols INTEGER,
    PRIMARY KEY (id, symbol)
);
CREATE INDEX IF NOT EXISTS news_symbol_ts ON news (symbol, created_at);
"""


class NewsStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def save(self, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        recs = [
            (int(r.id), r.symbol, pd.Timestamp(r.created_at).isoformat(), r.headline, r.source, int(r.n_symbols))
            for r in rows.itertuples(index=False)
        ]
        self.conn.executemany("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?)", recs)
        self.conn.commit()
        return len(recs)

    def load(self, symbols: list[str] | None = None) -> pd.DataFrame:
        q = "SELECT id, symbol, created_at, headline, source, n_symbols FROM news"
        params: list = []
        if symbols:
            q += f" WHERE symbol IN ({','.join('?' * len(symbols))})"
            params = list(symbols)
        df = pd.read_sql_query(q, self.conn, params=params)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
        return df

    def latest(self) -> datetime | None:
        row = self.conn.execute("SELECT MAX(created_at) FROM news").fetchone()
        return pd.Timestamp(row[0]).to_pydatetime() if row and row[0] else None


def fetch_alpaca_news(
    client, symbols: list[str], start: datetime, end: datetime, store: NewsStore,
    symbols_per_request: int = 20, window_days: int = 31,
) -> int:
    """Fetch news window by window (oldest first) so an interrupted run can resume."""
    from alpaca.data.requests import NewsRequest

    total = 0
    t = start
    while t < end:
        t_end = min(t + timedelta(days=window_days), end)
        n_window = 0
        for i in range(0, len(symbols), symbols_per_request):
            chunk = symbols[i : i + symbols_per_request]
            wanted = set(chunk)
            try:
                res = client.get_news(NewsRequest(symbols=",".join(chunk), start=t, end=t_end, include_content=False))
            except Exception as exc:  # keep going; a rerun fills gaps (INSERT OR IGNORE)
                log.warning("news request failed for %s..%s: %s", t.date(), t_end.date(), exc)
                continue
            rows = []
            for a in res.data.get("news", []):
                syms = list(a.symbols or [])
                for s in syms:
                    if s in wanted:
                        rows.append((a.id, s, a.created_at, a.headline, a.source, len(syms)))
            df = pd.DataFrame(rows, columns=["id", "symbol", "created_at", "headline", "source", "n_symbols"])
            n_window += store.save(df)
        total += n_window
        log.info("news %s -> %s: %d article-symbol rows", t.date(), t_end.date(), n_window)
        t = t_end
    return total


# --------------------------------------------------------------------------- features


def align_to_sessions(news: pd.DataFrame, market: Market, days: list[date]) -> pd.Series:
    """Map each article to the first trading day whose CLOSE is at or after publication. NaT if none."""
    closes = pd.DatetimeIndex([session_bounds(market, d)[1] for d in days]).tz_convert("UTC")
    idx = closes.searchsorted(news["created_at"].dt.tz_convert("UTC"), side="left")
    out = pd.Series([days[i] if i < len(days) else None for i in idx], index=news.index, dtype=object)
    return out


def daily_news_table(news: pd.DataFrame, market: Market, days: list[date]) -> pd.DataFrame:
    """One row per (session_date, symbol) with raw 1-day counts and scores."""
    if news.empty:
        return pd.DataFrame(columns=["session_date", "symbol", "news_count_1d", "sent_1d", "event_1d"])
    n = news.copy()
    n["session_date"] = align_to_sessions(n, market, days)
    n = n.dropna(subset=["session_date"])
    scores = n["headline"].map(score_headline)
    n["sent"] = [s for s, _ in scores]
    n["event"] = [e for _, e in scores]
    w = 1.0 / n["n_symbols"].clip(lower=1)
    focused = n["n_symbols"] <= MAX_SYMBOLS_FOR_SENTIMENT
    n["w_count"] = w
    n["w_sent"] = np.where(focused, n["sent"] * w, 0.0)
    n["w_event"] = np.where(focused, n["event"] * w, 0.0)
    g = n.groupby(["session_date", "symbol"])
    return pd.DataFrame(
        {"news_count_1d": g["w_count"].sum(), "sent_1d": g["w_sent"].sum(), "event_1d": g["w_event"].sum()}
    ).reset_index()


def add_news_features(feats: pd.DataFrame, news: pd.DataFrame, market: Market, warmup_days: int = 60) -> pd.DataFrame:
    """Merge news features into a daily feature frame. Rows before news coverage (+warmup) get NaN."""
    days = sorted(feats["session_date"].unique())
    table = daily_news_table(news, market, days)
    df = feats.merge(table, on=["session_date", "symbol"], how="left")
    for c in ("news_count_1d", "sent_1d", "event_1d"):
        df[c] = df[c].fillna(0.0)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["sent_5d"] = g["sent_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    df["event_5d"] = g["event_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    count_5d = g["news_count_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    norm_5d = g["news_count_1d"].transform(lambda s: s.shift(5).rolling(60, min_periods=20).mean()) * 5
    df["news_abn_5d"] = count_5d / (norm_5d + 1.0)

    if news.empty:
        df[NEWS_FEATURES] = np.nan
        return df
    start = news["created_at"].min().tz_convert(SESSIONS[market].tz).date()
    covered = [d for d in days if d >= start]
    cutoff = covered[min(warmup_days, len(covered) - 1)] if covered else None
    if cutoff is None:
        df[NEWS_FEATURES] = np.nan
    else:
        df.loc[df["session_date"] < cutoff, NEWS_FEATURES] = np.nan
    df[NEWS_FEATURES] = df[NEWS_FEATURES].astype("float32")
    return df
