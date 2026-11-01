# tradebot

An intraday algorithmic trading system for **US and ASX equities**, organised as a pipeline:

```
data → features → scanner → strategy (prediction) → risk manager → broker → journal
```

**Paper trading is the default.** To trade live you have to change the config *and* set a confirmation environment variable (see [Safety](#safety)).

> This is a research and engineering project, not financial advice. Most intraday strategies lose money once costs are counted. The backtester is built to show you that honestly.

---

## Quick start (no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q

# Walk-forward backtest on synthetic data with NO predictable structure.
# Expect zero trades or a small loss. That's the system working correctly.
tradebot backtest --synthetic --strategy ml

# Plant an edge to confirm the pipeline can find one
tradebot backtest --synthetic --strategy ml --momentum 0.4
```

Equity curves and trade logs are written to `reports/`.

### Diagnose before you tune

A backtest with zero trades can't tell "no signal" apart from "real signal smaller than costs". `diagnose` scores the walk-forward predictions on every bar:

```bash
tradebot diagnose --markets US --strategy ml
tradebot diagnose --markets US --strategy ml --horizon 12   # 60-minute horizon
```

- **IC t-stat:** below about 2 in absolute value means the predictions are indistinguishable from noise.
- **Decile table:** if the top decile's realised return is below the round-trip cost, the edge isn't tradeable at this frequency.

## Architecture

| Module | Role |
|---|---|
| `data/` | `DataProvider` interface; Alpaca (US), IBKR (ASX), synthetic; SQLite `BarStore` |
| `markets.py` | Session hours and timezones (New York / Sydney, DST-aware) |
| `features.py` | Returns, volatility, relative volume, VWAP distance, RSI, cross-sectional context; forward-return labels |
| `scanner.py` | Filters the universe on price, liquidity and relative volume; keeps the top N |
| `strategy/` | `momentum` baseline (empirically calibrated) and `ml` (gradient-boosted classifier + regressor) |
| `risk.py` | Sizing, position caps, gross exposure, edge-after-costs gate, stop / take-profit / time exits, end-of-day flatten, daily loss kill switch |
| `costs.py` | Commission (per-share, bps, minimum), half-spread, slippage |
| `execution/` | `SimBroker` (backtests), `AlpacaBroker`, `IBKRBroker` |
| `backtest.py` | Bar-by-bar walk-forward backtester using the same scanner, risk and cost code as live |
| `engine.py` | Live/paper loop: one `MarketRunner` per market, acting on each closed bar |
| `journal.py` | SQLite log of signals, orders and equity, for evaluating the model live |

### Rules that protect against fooling yourself

- **No look-ahead.** Features at bar *t* use only data up to *t*'s close. `tests/test_features.py` checks this by truncating the data.
- **Realistic fills.** A decision at bar *t*'s close fills at bar *t+1*'s open, after costs.
- **Walk-forward training.** The model only ever trains on days before the one it trades.
- **Labels match execution.** The label is the return from the next bar's open to the close *h* bars later, and it never spans two sessions.
- **Only trade a positive edge.** An entry needs `expected_return − round-trip costs ≥ min_edge_bps`.
- **Out-of-sample scoring.** Metrics and benchmarks cover only the period the model actually traded.

## What the first runs already show

1. **On pure noise, the ML model overfits and loses money to costs.** The momentum baseline correctly declines to trade. Any real-data result has to beat this.
2. **ASX minimum commissions are a wall for small accounts.** A $1,000 position with a $6 minimum commission costs about 1.2% round trip, far more than any plausible intraday edge. With a $10k AUD sleeve the risk manager will refuse nearly every ASX trade. The fix is larger positions, a lower-cost pricing plan, or longer holding periods.

## Real data setup

### US: Alpaca

1. Create an Alpaca account; paper trading is available on sign-up.
2. Copy `.env.example` to `.env` and add your **paper** API keys.
3. `pip install -e ".[alpaca]"`
4. Fetch data and backtest:

   ```bash
   tradebot fetch --markets US --days 60
   tradebot backtest --markets US
   ```

The free `iex` feed covers only the IEX exchange, so its volumes are a fraction of consolidated volume. Use `sip` (paid) for serious work.

Live trading as a non-US resident needs a separate application. Check with Alpaca that Australian residents are eligible.

### ASX: Interactive Brokers

1. Open an IBKR account and enable the paper trading account.
2. Subscribe to ASX market data. Historical intraday data requires it.
3. Run TWS or IB Gateway, logged into **paper**, with the API enabled.
4. `pip install -e ".[ibkr]"`, then:

   ```bash
   tradebot fetch --markets ASX --days 30
   ```

| Port | Session |
|---|---|
| 7497 | TWS paper |
| 4002 | Gateway paper |
| 7496 | TWS live (refused unless live trading is confirmed) |
| 4001 | Gateway live (refused unless live trading is confirmed) |

IBKR rate-limits historical data requests. Large universes will be slow to fetch.

### Running the loop

```bash
tradebot paper              # both markets
tradebot paper --markets US
```

The engine wakes on each closed bar (with a 20 s data delay) and retrains every `retrain_every_days`. It flattens positions before each close. On Ctrl-C it lists open positions but does not close them.

## Safety

- `mode: paper` is the default. `tradebot live` needs `mode: live` in the config **and** `TRADEBOT_CONFIRM_LIVE=I_UNDERSTAND_THE_RISKS` in the environment.
- Alpaca uses its paper endpoint unless live trading is confirmed.
- IBKR live ports are refused unless live trading is confirmed.
- Sizing never uses more than each market's `starting_capital` sleeve, even if the account holds more.
- A daily loss limit stops new entries and flattens all positions.
- The strategy cannot change itself during the day. Retraining happens only at the start of a day, on a schedule.

### US day-trading rules

The SEC approved scrapping FINRA's $25,000 pattern day trader minimum on 14 April 2026, effective 4 June 2026. Brokers have until 20 October 2027 to implement it, so check how yours applies intraday margin before trading frequently.

## Configuration

All settings live in `config/default.yaml`:

- universe, costs, capital sleeve and liquidity filters per market
- strategy horizon and training windows
- scanner settings
- risk limits

The cost numbers are **assumptions**. Replace them with your broker's actual pricing.

## Roadmap

- [ ] Exchange holiday and half-day calendars (e.g. `pandas_market_calendars`)
- [ ] Larger universes (ASX 200 / S&P 500) loaded from files
- [ ] Probability calibration and reliability plots for the ML model
- [ ] Purged / embargoed cross-validation for model selection
- [ ] Reconciliation of broker fills against the journal; slippage measured vs the model
- [ ] Streaming bars (Alpaca websocket, IBKR `keepUpToDate`) instead of polling
- [ ] Short selling (US only at first; ASX retail shorting is restricted)
- [ ] Combined AUD reporting across both markets (FX)
- [ ] Dashboard over the journal
- [ ] Cloud deployment (systemd / container) with alerting
