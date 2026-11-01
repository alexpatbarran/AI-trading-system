"""Command line interface.

  tradebot backtest --synthetic                 # offline, no API keys needed
  tradebot backtest --synthetic --momentum 0.1  # with a planted edge (sanity check)
  tradebot fetch --days 60                      # download bars into the local store
  tradebot backtest                             # backtest on stored real data
  tradebot paper                                # run against paper accounts
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
from tradebot.strategy import make_strategy


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
    for k in ["sharpe_annualised", "profit_factor", "trades", "trades_per_day", "final_equity"]:
        print(f"  {k:24s} {m[k]:.2f}")
    if r.universe_buy_hold is not None:
        print(f"  {'universe_buy_hold':24s} {r.universe_buy_hold:+.2%}")
    if r.benchmark_buy_hold is not None:
        print(f"  {'benchmark_buy_hold':24s} {r.benchmark_buy_hold:+.2%}")
    if r.risk_rejections:
        print(f"  risk rejections: {r.risk_rejections}")


def cmd_backtest(args, settings: Settings) -> int:
    if args.strategy:
        settings.strategy.name = args.strategy
    out = Path("reports")
    out.mkdir(exist_ok=True)
    for market in _markets(settings, args.markets):
        mcfg = settings.markets[market]
        symbols = mcfg.universe + ([mcfg.benchmark] if mcfg.benchmark else [])
        if args.synthetic:
            from tradebot.data.synthetic import generate_bars, trading_days

            end = datetime.now(timezone.utc).date() - timedelta(days=1)
            days = trading_days(market, end, args.days, set(mcfg.holidays))
            bars = generate_bars(market, symbols, days, settings.bar_minutes, momentum=args.momentum, seed=args.seed)
        else:
            from tradebot.data.store import BarStore

            bars = BarStore(settings.db_path).load(market, settings.bar_minutes, symbols)
            if bars.empty:
                print(f"No stored bars for {market.value}. Run `tradebot fetch` first or use --synthetic.")
                continue
        result = Backtester(settings, market, make_strategy(settings.strategy, seed=args.seed)).run(bars)
        _print_result(result)
        tag = f"{market.value}_{result.strategy}"
        result.equity.to_csv(out / f"equity_{tag}.csv")
        result.trades.to_csv(out / f"trades_{tag}.csv", index=False)
    return 0


def cmd_fetch(args, settings: Settings) -> int:
    from tradebot.data.store import BarStore
    from tradebot.wiring import build_data

    store = BarStore(settings.db_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    for market in _markets(settings, args.markets):
        mcfg = settings.markets[market]
        symbols = mcfg.universe + ([mcfg.benchmark] if mcfg.benchmark else [])
        bars = build_data(settings, market).get_bars(symbols, start, end, settings.bar_minutes)
        n = store.save(market, settings.bar_minutes, bars)
        print(f"{market.value}: stored {n} bars for {bars['symbol'].nunique() if n else 0} symbols")
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

    b = sub.add_parser("backtest", help="walk-forward backtest")
    b.add_argument("--synthetic", action="store_true", help="use generated data (no API keys)")
    b.add_argument("--days", type=int, default=60, help="synthetic trading days")
    b.add_argument("--momentum", type=float, default=0.0, help="planted autocorrelation for synthetic data")
    b.add_argument("--strategy", choices=["momentum", "ml"])
    b.add_argument("--markets", nargs="+", choices=["US", "ASX"])
    b.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("fetch", help="download historical bars into the local store")
    f.add_argument("--days", type=int, default=60)
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
    if args.cmd == "fetch":
        return cmd_fetch(args, settings)
    return cmd_run(args, settings, live=args.cmd == "live")


if __name__ == "__main__":
    sys.exit(main())
