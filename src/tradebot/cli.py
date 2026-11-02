"""Command line interface.

  tradebot backtest --synthetic                 # offline, no API keys needed
  tradebot backtest --synthetic --momentum 0.1  # with a planted edge (sanity check)
  tradebot fetch --days 60                      # download bars into the local store
  tradebot backtest                             # backtest on stored real data
  tradebot diagnose                             # does the model predict anything?
  tradebot fetch --timeframe daily              # ~10 years of daily bars
  tradebot diagnose --timeframe daily --strategy reversal
  tradebot paper                                # run against paper accounts (intraday)
  tradebot live                                 # real money; needs explicit confirmation
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradebot.backtest import Backtester, BacktestResult
from tradebot.config import SIZE_BANDS, Settings, assert_live_allowed, load_settings
from tradebot.domain import Market
from tradebot.strategy import DAILY_STRATEGIES, INTRADAY_STRATEGIES, make_daily_strategy, make_strategy


def _markets(settings: Settings, names: list[str] | None) -> list[Market]:
    return [Market(n) for n in names] if names else settings.enabled_markets()


def _print_result(r: BacktestResult) -> None:
    m = r.metrics
    print(f"\n=== {r.market.value} | strategy={r.strategy} | out-of-sample days={r.days_traded} ===")
    if not m:
        print("  no results (not enough data to train?)")
        return
    for k in ["total_return", "max_drawdown", "win_rate", "avg_trade_return", "costs_pct_of_capital"]:
        print(f"  {k:24s} {m[k]:+.2%}")
    for k in ["sharpe_annualised", "profit_factor", "trades", "trades_per_day", "avg_holding_days", "final_equity"]:
        if k in m:
            print(f"  {k:24s} {m[k]:.2f}")
    if r.universe_buy_hold is not None:
        print(f"  {'universe_buy_hold':24s} {r.universe_buy_hold:+.2%}")
    if r.benchmark_buy_hold is not None:
        print(f"  {'benchmark_buy_hold':24s} {r.benchmark_buy_hold:+.2%}")
    if r.risk_rejections:
        print(f"  risk rejections: {r.risk_rejections}")


DAILY = 1440  # bar size key used for daily bars in the store


def _daily(args) -> bool:
    return getattr(args, "timeframe", "intraday") == "daily"


def _which(args) -> str:
    return getattr(args, "universe", "default") or "default"


def _symbols(settings: Settings, market: Market, daily: bool, which: str = "default") -> list[str]:
    mcfg = settings.markets[market]
    try:
        symbols = settings.universe(market, "daily" if daily else "intraday", which)
    except FileNotFoundError:
        raise SystemExit(f"{market.value}: no broad universe file yet. Run `tradebot build-universe` first.")
    return symbols + ([mcfg.benchmark] if mcfg.benchmark and mcfg.benchmark not in symbols else [])


def _load_bars(args, settings: Settings, market: Market):
    mcfg = settings.markets[market]
    daily = _daily(args)
    symbols = _symbols(settings, market, daily, "default" if args.synthetic else _which(args))
    if args.synthetic:
        from tradebot.data.synthetic import generate_bars, generate_daily_bars, trading_days

        n_days = args.days or (1500 if daily else 60)
        end = datetime.now(timezone.utc).date() - timedelta(days=1)
        days = trading_days(market, end, n_days, set(mcfg.holidays))
        if daily:
            return generate_daily_bars(market, symbols, days, momentum=args.momentum, seed=args.seed)
        return generate_bars(market, symbols, days, settings.bar_minutes, momentum=args.momentum, seed=args.seed)
    from tradebot.data.store import BarStore

    bar_key = DAILY if daily else settings.bar_minutes
    bars = BarStore(settings.db_path).load(market, bar_key, symbols)
    if bars.empty:
        tf = " --timeframe daily" if daily else ""
        print(f"No stored bars for {market.value}. Run `tradebot fetch{tf}` first or use --synthetic.")
        return None
    return bars


def _apply_overrides(args, settings: Settings) -> None:
    daily = _daily(args)
    valid = DAILY_STRATEGIES if daily else INTRADAY_STRATEGIES
    if args.strategy and args.strategy not in valid:
        raise SystemExit(f"--strategy {args.strategy} isn't available for this timeframe; choose from {valid}")
    if args.strategy and not daily:
        settings.strategy.name = args.strategy
    if args.horizon:
        if daily:
            settings.daily.horizon_days = args.horizon
        else:
            settings.strategy.horizon_bars = args.horizon
    if (args.news or args.size != "all" or _which(args) != "default") and not daily:
        raise SystemExit("--news, --size and --universe apply to --timeframe daily only")
    settings.daily.size_min_rank, settings.daily.size_max_rank = SIZE_BANDS[args.size]


def _make(args, settings: Settings):
    if _daily(args):
        return make_daily_strategy(args.strategy or "ml", seed=args.seed, news=args.news)
    return make_strategy(settings.strategy, seed=args.seed)


def _load_news(args, settings: Settings, market: Market, bars):
    if not args.news:
        return None
    from tradebot.news import NewsStore

    news = NewsStore(settings.db_path).load(sorted(bars["symbol"].unique()))
    if news.empty:
        raise SystemExit(f"No stored news for {market.value}. Run `tradebot fetch-news` first.")
    print(f"{market.value}: {len(news)} article-symbol rows, {news['created_at'].min():%Y-%m-%d} to "
          f"{news['created_at'].max():%Y-%m-%d}")
    return news


def _label(args) -> str:
    parts = []
    if _which(args) != "default":
        parts.append(f"universe={_which(args)}")
    if args.size != "all":
        parts.append(f"size={args.size}")
    if args.news:
        parts.append("news")
    return f"[{', '.join(parts)}]" if parts else ""


def cmd_diagnose(args, settings: Settings) -> int:
    from tradebot.research import diagnose, diagnose_daily, format_diagnosis

    _apply_overrides(args, settings)
    for market in _markets(settings, args.markets):
        bars = _load_bars(args, settings, market)
        if bars is None:
            continue
        strategy = _make(args, settings)
        if _daily(args):
            d = diagnose_daily(settings, market, strategy, bars, _load_news(args, settings, market, bars))
        else:
            d = diagnose(settings, market, strategy, bars)
        if _label(args):
            print(f"\n{_label(args)}", end="")
        print(format_diagnosis(d) if d else f"\n{market.value}: not enough data to train "
              "(too little history, or no stocks in this --size band for this universe)")
    return 0


def cmd_features(args, settings: Settings) -> int:
    from tradebot.research import feature_scan, format_feature_scan

    args.timeframe = "daily"
    args.strategy = None
    _apply_overrides(args, settings)
    for market in _markets(settings, args.markets):
        bars = _load_bars(args, settings, market)
        if bars is None:
            continue
        df = feature_scan(settings, market, bars, _load_news(args, settings, market, bars))
        print(f"\n=== {market.value} {_label(args)} | feature scan | horizon={settings.daily.horizon_days} days ===")
        print(format_feature_scan(df))
    return 0


def cmd_backtest(args, settings: Settings) -> int:
    from tradebot.daily_backtest import DailyBacktester

    _apply_overrides(args, settings)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    for market in _markets(settings, args.markets):
        bars = _load_bars(args, settings, market)
        if bars is None:
            continue
        if _daily(args):
            news = _load_news(args, settings, market, bars)
            result = DailyBacktester(settings, market, _make(args, settings), news).run(bars)
        else:
            result = Backtester(settings, market, _make(args, settings)).run(bars)
        if _label(args):
            print(f"\n{_label(args)}", end="")
        _print_result(result)
        tag = f"{market.value}_{'daily_' if _daily(args) else ''}{result.strategy.split(' ')[0]}"
        extras = [_which(args) if _which(args) != "default" else "", args.size if args.size != "all" else ""]
        extras.append("news" if args.news else "")
        tag += "".join(f"_{e}" for e in extras if e)
        result.equity.to_csv(out / f"equity_{tag}.csv")
        result.trades.to_csv(out / f"trades_{tag}.csv", index=False)
    return 0


def cmd_fetch(args, settings: Settings) -> int:
    from tradebot.data.store import BarStore
    from tradebot.wiring import build_data

    daily = _daily(args)
    if _which(args) != "default" and not daily:
        raise SystemExit("--universe broad applies to --timeframe daily only")
    store = BarStore(settings.db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days or (3650 if daily else 60))
    bar_key = DAILY if daily else settings.bar_minutes
    for market in _markets(settings, args.markets):
        symbols = _symbols(settings, market, daily, _which(args))
        print(f"{market.value}: fetching {len(symbols)} symbols ({'daily' if daily else f'{bar_key}-min'} bars)...")
        bars = build_data(settings, market).get_bars(symbols, start, end, bar_key)
        n = store.save(market, bar_key, bars)
        got = bars["symbol"].nunique() if n else 0
        print(f"{market.value}: stored {n} bars for {got}/{len(symbols)} symbols")
        missing = sorted(set(symbols) - set(bars["symbol"])) if n else symbols
        if missing:
            shown = " ".join(missing[:40]) + (f" ... (+{len(missing) - 40} more)" if len(missing) > 40 else "")
            print(f"  missing: {shown}")
    return 0


def cmd_build_universe(args, settings: Settings) -> int:
    import os

    from alpaca.trading.client import TradingClient

    from tradebot.data.alpaca import AlpacaData
    from tradebot.universe import build_us_universe, write_universe

    mcfg = settings.markets[Market.US]
    if not mcfg.broad_universe_file:
        raise SystemExit("Set markets.US.broad_universe_file in the config")
    trading = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True, raw_data=True)
    data = AlpacaData(feed=settings.alpaca.feed, daily_feed=settings.alpaca.daily_feed)
    symbols = build_us_universe(data, trading, args.years, args.top, args.min_months, args.min_price)
    note = (f"top {args.top} by monthly traded value in >= {args.min_months} months of the last {args.years} "
            f"years, price >= ${args.min_price:g}; includes delisted stocks where Alpaca has data")
    path = mcfg.broad_universe_file
    from tradebot.config import REPO_ROOT

    target = path if os.path.isabs(path) else REPO_ROOT / path
    write_universe(target, symbols, note)
    print(f"Wrote {len(symbols)} symbols to {target}")
    print("Next: tradebot fetch --timeframe daily --universe broad --markets US")
    return 0


def cmd_fetch_news(args, settings: Settings) -> int:
    import os

    from alpaca.data.historical.news import NewsClient

    from tradebot.news import NewsStore, fetch_alpaca_news

    store = NewsStore(settings.db_path)
    symbols = [s for s in _symbols(settings, Market.US, True, _which(args)) if s != settings.markets[Market.US].benchmark]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    latest = store.latest()
    if latest and not args.restart and latest > start:
        start = datetime(latest.year, latest.month, 1, tzinfo=timezone.utc)  # resume from the last month stored
        print(f"Resuming from {start:%Y-%m-%d} (use --restart to refetch everything)")
    print(f"Fetching news for {len(symbols)} symbols, {start:%Y-%m-%d} to {end:%Y-%m-%d}. This can take a while.")
    client = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    n = fetch_alpaca_news(client, symbols, start, end, store)
    print(f"Stored {n} article-symbol rows")
    return 0


def cmd_run(args, settings: Settings, live: bool) -> int:
    from tradebot.engine import Engine, MarketRunner
    from tradebot.journal import Journal
    from tradebot.wiring import build_broker, build_data

    if live:
        assert_live_allowed(settings)
        print("!!! LIVE TRADING WITH REAL MONEY !!!")
    else:
        settings.mode = "paper"
    journal = Journal(settings.db_path)
    runners = [
        MarketRunner(
            settings, m, build_data(settings, m), build_broker(settings, m),
            make_strategy(settings.strategy), journal,
        )
        for m in _markets(settings, args.markets)
    ]
    Engine(runners).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tradebot")
    p.add_argument("--config", help="path to YAML config (default: config/default.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_ in (
        ("backtest", "walk-forward backtest"),
        ("diagnose", "does the model predict anything?"),
        ("features", "daily: which individual features predict returns? (exploratory)"),
    ):
        b = sub.add_parser(name, help=help_)
        b.add_argument("--timeframe", choices=["intraday", "daily"], default="intraday")
        b.add_argument("--synthetic", action="store_true", help="use generated data (no API keys)")
        b.add_argument("--days", type=int, help="synthetic trading days (default 60 intraday / 1500 daily)")
        b.add_argument(
            "--momentum", type=float, default=0.0,
            help="planted autocorrelation for synthetic data (negative = reversal)",
        )
        b.add_argument("--strategy", choices=sorted(set(INTRADAY_STRATEGIES + DAILY_STRATEGIES)))
        b.add_argument("--horizon", type=int, help="bars (intraday) or days (daily) to hold/predict")
        b.add_argument("--markets", nargs="+", choices=["US", "ASX"])
        b.add_argument("--seed", type=int, default=0)
        b.add_argument("--universe", choices=["default", "broad"], default="default",
                       help="daily: the large-cap list, or the broad list from build-universe")
        b.add_argument("--size", choices=list(SIZE_BANDS), default="all",
                       help="daily: point-in-time size band by traded-value rank")
        b.add_argument("--news", action="store_true", help="daily: add news features (run fetch-news first)")

    f = sub.add_parser("fetch", help="download historical bars into the local store")
    f.add_argument("--timeframe", choices=["intraday", "daily"], default="intraday")
    f.add_argument("--days", type=int, help="calendar days of history (default 60 intraday / 3650 daily)")
    f.add_argument("--markets", nargs="+", choices=["US", "ASX"])
    f.add_argument("--universe", choices=["default", "broad"], default="default")

    u = sub.add_parser("build-universe", help="build a broad US universe incl. delisted stocks (Alpaca)")
    u.add_argument("--years", type=int, default=6)
    u.add_argument("--top", type=int, default=1500, help="keep stocks ranked in the top N by traded value...")
    u.add_argument("--min-months", type=int, default=3, help="...in at least this many months")
    u.add_argument("--min-price", type=float, default=5.0)

    n = sub.add_parser("fetch-news", help="download Alpaca news headlines for the US universe")
    n.add_argument("--days", type=int, default=1825, help="calendar days of history (default ~5 years)")
    n.add_argument("--universe", choices=["default", "broad"], default="default")
    n.add_argument("--restart", action="store_true", help="ignore stored news and refetch from the start")

    for name in ("paper", "live"):
        r = sub.add_parser(name, help=f"run the trading loop ({name})")
        r.add_argument("--markets", nargs="+", choices=["US", "ASX"])

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    if args.cmd == "backtest":
        return cmd_backtest(args, settings)
    if args.cmd == "diagnose":
        return cmd_diagnose(args, settings)
    if args.cmd == "features":
        return cmd_features(args, settings)
    if args.cmd == "fetch":
        return cmd_fetch(args, settings)
    if args.cmd == "build-universe":
        return cmd_build_universe(args, settings)
    if args.cmd == "fetch-news":
        return cmd_fetch_news(args, settings)
    return cmd_run(args, settings, live=args.cmd == "live")


if __name__ == "__main__":
    sys.exit(main())
