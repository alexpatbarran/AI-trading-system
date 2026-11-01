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
from tradebot.config import Settings, assert_live_allowed, load_settings
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


def _symbols(settings: Settings, market: Market, daily: bool) -> list[str]:
    mcfg = settings.markets[market]
    symbols = settings.universe(market, "daily" if daily else "intraday")
    return symbols + ([mcfg.benchmark] if mcfg.benchmark and mcfg.benchmark not in symbols else [])


def _load_bars(args, settings: Settings, market: Market):
    mcfg = settings.markets[market]
    daily = _daily(args)
    symbols = _symbols(settings, market, daily)
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


def _make(args, settings: Settings):
    if _daily(args):
        return make_daily_strategy(args.strategy or "ml", seed=args.seed)
    return make_strategy(settings.strategy, seed=args.seed)


def cmd_diagnose(args, settings: Settings) -> int:
    from tradebot.research import diagnose, diagnose_daily, format_diagnosis

    _apply_overrides(args, settings)
    fn = diagnose_daily if _daily(args) else diagnose
    for market in _markets(settings, args.markets):
        bars = _load_bars(args, settings, market)
        if bars is None:
            continue
        d = fn(settings, market, _make(args, settings), bars)
        print(format_diagnosis(d) if d else f"\n{market.value}: not enough data to train")
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
        cls = DailyBacktester if _daily(args) else Backtester
        result = cls(settings, market, _make(args, settings)).run(bars)
        _print_result(result)
        tag = f"{market.value}_{'daily_' if _daily(args) else ''}{result.strategy.split(' ')[0]}"
        result.equity.to_csv(out / f"equity_{tag}.csv")
        result.trades.to_csv(out / f"trades_{tag}.csv", index=False)
    return 0


def cmd_fetch(args, settings: Settings) -> int:
    from tradebot.data.store import BarStore
    from tradebot.wiring import build_data

    daily = _daily(args)
    store = BarStore(settings.db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days or (3650 if daily else 60))
    bar_key = DAILY if daily else settings.bar_minutes
    for market in _markets(settings, args.markets):
        symbols = _symbols(settings, market, daily)
        print(f"{market.value}: fetching {len(symbols)} symbols ({'daily' if daily else f'{bar_key}-min'} bars)...")
        bars = build_data(settings, market).get_bars(symbols, start, end, bar_key)
        n = store.save(market, bar_key, bars)
        got = bars["symbol"].nunique() if n else 0
        print(f"{market.value}: stored {n} bars for {got}/{len(symbols)} symbols")
        missing = sorted(set(symbols) - set(bars["symbol"])) if n else symbols
        if missing:
            print(f"  missing: {' '.join(missing)}")
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

    for name, help_ in (("backtest", "walk-forward backtest"), ("diagnose", "does the model predict anything?")):
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

    f = sub.add_parser("fetch", help="download historical bars into the local store")
    f.add_argument("--timeframe", choices=["intraday", "daily"], default="intraday")
    f.add_argument("--days", type=int, help="calendar days of history (default 60 intraday / 3650 daily)")
    f.add_argument("--markets", nargs="+", choices=["US", "ASX"])

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
    if args.cmd == "fetch":
        return cmd_fetch(args, settings)
    return cmd_run(args, settings, live=args.cmd == "live")


if __name__ == "__main__":
    sys.exit(main())
