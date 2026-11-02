# Searching for an Equity Trading Edge

Alexander Barranquero · November 2026

Across 17 walk-forward tests on US stocks, from 5-minute bars to monthly holds, only one model found a reliable signal: in small caps, it spots stocks that will lag. After realistic trading costs, no long-only strategy beat simply holding the universe.

## Results at a glance

Each row is one test's IC t-stat: how consistently its stock rankings beat chance, day after day. Below |t| = 2 is indistinguishable from luck. With 17 tests run, a result needs about t = 2.9 (Bonferroni) before it can be trusted.

| Stage | Test | IC t-stat |
| --- | --- | --- |
| Intraday, US mega caps | ML, 30 min (32 days) | +0.73 |
| | Momentum, 30 min (32 days) | −0.33 |
| | ML, 60 min (32 days) | −0.44 |
| | ML, 30 min (240 days) | +0.53 |
| Daily, US large caps | Reversal, 5 days | −0.19 |
| | Momentum 12-1, 5 days | −0.06 |
| | ML, 5 days | −0.47 |
| | Momentum 12-1, 21 days | −1.25 |
| | Reversal, 21 days | +0.83 |
| Daily, large caps + news | News rule, 5 days | −1.35 |
| | ML + news, 5 days | +0.34 |
| Daily, broad universe | Reversal, mid caps, 5 days | +1.06 |
| | ML, mid caps, 5 days | +1.31 |
| | Reversal, small caps, 5 days | +1.82 |
| | **ML, small caps, 5 days** | **+4.98** |
| | **ML, small caps, 10 days** | **+3.45** |
| | ML, small caps, 21 days | +2.71 |

## The one real signal: small caps

An ML model on small caps (stocks ranked 801st or lower by daily traded value) is the only test that clears the bar.

| Horizon | IC | t-stat | Worst decile vs average | Best decile vs average | Round-trip cost |
| --- | --- | --- | --- | --- | --- |
| 5 days | +0.038 | +4.98 | −14.1 bps | +4.5 bps | 19.9 bps |
| 10 days | +0.036 | +3.45 | −24.8 bps | +0.4 bps | 20.0 bps |
| 21 days | +0.038 | +2.71 | −51.8 bps | +20.3 bps | 19.9 bps |

The model is good at one thing: flagging small caps that will lag. Its worst-ranked decile lost more often than it won (hit rate about 45%), yet still averaged a positive return. That fits lottery-like stocks, which mostly drift down but now and then jump. The best-ranked decile covers its cost only at 21 days, and then only just.

A per-feature scan (5-day horizon, small caps) shows what the model is picking up:

| Feature | IC | t-stat | Reading | Known effect |
| --- | --- | --- | --- | --- |
| Distance from 52-week high | +0.049 | +5.05 | Stocks near their high do better | 52-week-high effect |
| Daily high-low range | −0.041 | −4.64 | Wide-swinging stocks do worse | Lottery / low-volatility effect |
| 12-month momentum (skip 1 month) | +0.034 | +4.39 | Last year's winners keep winning | Momentum |
| 21-day volatility | −0.043 | −4.33 | Volatile stocks do worse | Idiosyncratic-volatility puzzle |
| Distance from 200-day average | +0.030 | +4.03 | Stocks in an uptrend do better | Trend |
| Volume vs normal | +0.008 | +3.28 | Unusual volume helps slightly | Attention |

These are textbook anomalies. They show up in small caps but not in large caps, as the research predicts: they persist where trading is expensive enough to stop arbitrageurs.

## How the system works

Every result comes from one pipeline, built so a strategy can only score well on data it could not have seen.

1. **Data:** Alpaca 5-minute and daily bars, adjusted for splits and dividends. Universes are 121 US large caps and a broad set of 2,475 stocks that includes delisted ones, plus 307,000 Benzinga headlines.
2. **Features:** returns, volatility, trend, distance from the 52-week high, volume, RSI and cross-sectional ranks. News adds attention, keyword sentiment and events such as upgrades and earnings beats.
3. **Model:** walk-forward training on the previous 2 years, retrained monthly, never on the day it trades. Baselines are rule-based reversal and momentum.
4. **Risk and costs:** position and exposure caps, stop-losses and a daily loss limit. Spreads are tiered by liquidity from 2 to 40 bps, plus 3 bps slippage per side.
5. **Execution:** decide at the close and fill at the next open. Paper trading by default, through Alpaca (US) or Interactive Brokers (ASX).
6. **Diagnostics:** cross-sectional IC with overlap-corrected t-stats, and decile tables compared with the average stock and with costs.

Safeguards against fooling ourselves:

- **No look-ahead:** tests recompute features on truncated data and require identical values.
- **Purged training:** rows whose labels overlap the trading day are dropped from training.
- **Overlap-corrected t-stats:** a Newey-West correction stops overlapping multi-day returns from inflating significance.
- **Same-day comparisons:** deciles rank stocks against each other on the same day, not good periods against bad ones.
- **Planted-edge checks:** the system must find edges planted in synthetic data, and find nothing in pure noise. On noise, ML t-stats averaged −0.06 over 10 seeds.
- **Live-trading lock:** real money needs a config change plus a confirmation variable. 37 automated tests run on every push.

## Bugs the checks caught

| Problem | What it would have done | Fix |
| --- | --- | --- |
| Deciles pooled across all days | Showed a fake 70 bps spread that sorted bull periods from bear ones, while IC said zero | Rank stocks within each day |
| Stray 2012 articles in a 2021+ news fetch | Taught the model that years of "no news" were real data | Start coverage at sustained news flow |
| Alpaca returns pre-market and after-hours bars | Let the system trade in thin early-morning markets | Keep regular-session bars only |
| ML scored slightly above zero on random data | Looked like leakage of future prices | Traced to a real beta effect in the synthetic data; with zero drift, t-stats centred on −0.06 |

## What it means, and the limits

For a small account, the evidence points to a low-cost index fund. The system's real value is that it can say so with numbers rather than hope.

- **Avoid the losers:** skipping the worst-ranked small-cap decile would have added about 5.8 bps per 21 days (roughly 0.7% a year), before rebalancing costs.
- **Short the losers:** the bigger gap, but small caps are often hard or expensive to borrow, and borrow costs are not modelled.

Limits:

- **Survivorship bias:** the large-cap list is today's winners. The broad universe includes delisted stocks only where Alpaca still holds their data.
- **Missing final losses:** a delisted stock's last move has no future close, so it drops out of the test. The bottom-decile effect is probably understated.
- **Assumed costs:** spreads and slippage are estimates, not measured fills.
- **Thin news model:** a keyword lexicon on headlines only, and entries at the next open miss the overnight reaction.
- **One market regime:** daily data covers about 2016–2026, mostly a bull market; intraday covers one year.
- **ASX untested:** it needs an Interactive Brokers data subscription. A $6 minimum commission also costs about 1.2% per round trip on a $1,000 position.

## Reproduce

```bash
tradebot fetch --timeframe daily --markets US
tradebot diagnose --timeframe daily --markets US --strategy ml
tradebot build-universe
tradebot fetch --timeframe daily --universe broad --markets US --days 2200
tradebot diagnose --timeframe daily --markets US --universe broad --size small --strategy ml
tradebot features --markets US --universe broad --size small
```

## Sources

- Momentum: Jegadeesh and Titman, [Returns to Buying Winners and Selling Losers](https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html), *Journal of Finance* 48(1), 1993.
- 52-week high: George and Hwang, [The 52-Week High and Momentum Investing](https://www.cxoadvisory.com/1284/technical-trading/the-52-week-high-as-a-momentum-indicator-for-individual-stocks/), *Journal of Finance*, 2004 (summary page).
- Volatility: Ang, Hodrick, Xing and Zhang, [The Cross-Section of Volatility and Expected Returns](https://papers.ssrn.com/abstract=681343) (SSRN working paper).
- Lottery stocks: Bali, Cakici and Whitelaw, [Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns](https://www.nber.org/papers/w14804.pdf) (NBER working paper 14804).
- Decay after publication: McLean and Pontiff, [Does Academic Research Destroy Stock Return Predictability?](https://counterpointfunds.com/wp-content/uploads/2017/07/PredictabilityMcleanPontiff.pdf), *Journal of Finance*. Across 97 predictors, returns fall about 58% after publication.
