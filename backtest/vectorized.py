import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


def run_backtest(prices_df, lookback=252, skip=21, rebalance_freq=21):
    """Vectorized backtest of the cross-sectional momentum strategy.

    At each rebalance date, compute the 12-1 momentum signal and assign
    equal weights to longs/shorts. Hold until next rebalance.

    Args:
        prices_df:      DataFrame of closing prices (assets as columns)
        lookback:       Momentum lookback in trading days (default 252)
        skip:           Days to skip at end of lookback (default 21)
        rebalance_freq: How often to rebalance in trading days (default 21 = monthly)

    Returns:
        results: dict with equity curve, returns, turnover, and per-period weights
    """
    from signals.momentum.cross_sectional import compute_momentum, compute_momentum_weights

    daily_returns = prices_df.pct_change()
    n_days = len(prices_df)
    assets = prices_df.columns.tolist()

    # We need at least lookback + skip days before first signal
    min_idx = lookback + skip

    # Generate rebalance dates (indices into prices_df)
    rebalance_indices = list(range(min_idx, n_days, rebalance_freq))

    # Store weights per period: index = rebalance date, columns = assets
    weights_history = []

    for idx in rebalance_indices:
        # Slice prices up to and including this rebalance date
        prices_slice = prices_df.iloc[: idx + 1]
        signal_df = compute_momentum(prices_slice, lookback=lookback, skip=skip)
        w = compute_momentum_weights(signal_df)
        weights_history.append({
            "date": prices_df.index[idx],
            **w.to_dict()
        })

    weights_df = pd.DataFrame(weights_history).set_index("date")

    # Forward-fill weights into daily index (hold until next rebalance)
    # Align weights to full daily returns index
    daily_weights = weights_df.reindex(daily_returns.index).ffill()

    # Shift weights by 1 day: signal on day T → trade at open day T+1
    daily_weights = daily_weights.shift(1).fillna(0)

    # Strategy daily return = sum of (weight_i * return_i) across assets
    strategy_returns = (daily_weights * daily_returns).sum(axis=1)

    # Trim to the period where we have actual signals
    first_signal_date = weights_df.index[0]
    strategy_returns = strategy_returns.loc[first_signal_date:]

    # Equity curve (starting at 1.0)
    equity_curve = (1 + strategy_returns).cumprod()

    # Benchmark: equal-weight buy-and-hold
    ew_returns = daily_returns.mean(axis=1).loc[first_signal_date:]
    ew_equity = (1 + ew_returns).cumprod()

    # Performance summary
    n = len(strategy_returns)
    ann_factor = 252

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(ann_factor)

    def max_drawdown(eq):
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        return dd.min()

    def cagr(eq):
        years = len(eq) / ann_factor
        return (eq.iloc[-1] ** (1 / years)) - 1

    summary = pd.DataFrame({
        "Momentum": [
            round(cagr(equity_curve) * 100, 2),
            round(sharpe(strategy_returns), 2),
            round(max_drawdown(equity_curve) * 100, 2),
            round(strategy_returns.std() * np.sqrt(ann_factor) * 100, 2),
        ],
        "Equal-Weight B&H": [
            round(cagr(ew_equity) * 100, 2),
            round(sharpe(ew_returns), 2),
            round(max_drawdown(ew_equity) * 100, 2),
            round(ew_returns.std() * np.sqrt(ann_factor) * 100, 2),
        ]
    }, index=["CAGR (%)", "Sharpe", "Max Drawdown (%)", "Ann. Vol (%)"])

    return {
        "equity_curve": equity_curve,
        "ew_equity": ew_equity,
        "strategy_returns": strategy_returns,
        "weights_df": weights_df,
        "summary": summary,
    }


# --- Run when called directly ---
if __name__ == "__main__":
    from data.loaders.equity import get_prices

    TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM"]
    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")  # More history for backtest

    print("Running backtest...")
    results = run_backtest(prices_df)

    print("\n=== Performance Summary ===")
    print(results["summary"])

    print("\n=== Equity Curve (last 5 rows) ===")
    print(results["equity_curve"].tail())

    print("\n=== Weight History (last 5 rebalances) ===")
    print(results["weights_df"].tail())