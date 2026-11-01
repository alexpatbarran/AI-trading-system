"""Performance metrics. Judge a strategy on these, after costs, vs a benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    starting_capital: float,
    tz,
    total_commission: float = 0.0,
    total_spread_slippage: float = 0.0,
) -> dict:
    if equity.empty:
        return {}
    local = equity.copy()
    local.index = local.index.tz_convert(tz)
    daily = local.groupby(local.index.date).last()
    daily_ret = pd.concat([pd.Series([starting_capital]), daily.reset_index(drop=True)]).pct_change().dropna()

    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_loss = -losses["pnl"].sum() if n else 0.0

    return {
        "days": len(daily),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / starting_capital - 1.0),
        "sharpe_annualised": sharpe,
        "max_drawdown": float(drawdown.min()),
        "trades": n,
        "trades_per_day": n / max(len(daily), 1),
        "win_rate": float(len(wins) / n) if n else 0.0,
        "avg_trade_return": float(trades["return"].mean()) if n else 0.0,
        "profit_factor": float(wins["pnl"].sum() / gross_loss) if n and gross_loss > 0 else float("nan"),
        "commission_paid": total_commission,
        "spread_slippage_paid": total_spread_slippage,
        "costs_pct_of_capital": (total_commission + total_spread_slippage) / starting_capital,
    }


def buy_and_hold_return(bars: pd.DataFrame, symbol: str | None = None) -> float | None:
    """Equal-weight buy-and-hold over the period (or a single benchmark symbol)."""
    df = bars if symbol is None else bars[bars["symbol"] == symbol]
    if df.empty:
        return None
    df = df.sort_values("timestamp")
    first = df.groupby("symbol")["open"].first()
    last = df.groupby("symbol")["close"].last()
    return float((last / first - 1.0).mean())
