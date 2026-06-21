import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


def run_pairs_backtest(prices_df, pairs=None, hedge_window=60,
                       zscore_window=20, entry_threshold=1.5,
                       exit_threshold=0.25, vol_target=0.10):
    """Backtest all pairs with volatility-scaled position sizing.

    Each pair's position size is scaled so that all pairs contribute
    equal volatility to the portfolio (vol targeting). This prevents
    high-vol pairs like GLD/SLV from dominating low-vol pairs like TLT/IEF.

    Args:
        vol_target: Target annualized volatility per pair (default 10%)
    """
    from signals.mean_reversion.pairs_trading import compute_pair_signal, PAIRS

    if pairs is None:
        pairs = PAIRS

    daily_returns = prices_df.pct_change()
    pair_return_series = {}

    for asset_a, asset_b in pairs:
        if asset_a not in prices_df.columns or asset_b not in prices_df.columns:
            continue

        pair_name = f"{asset_a}/{asset_b}"
        print(f"  Backtesting pair: {pair_name}")

        signals = compute_pair_signal(
            prices_df[asset_a], prices_df[asset_b],
            hedge_window=hedge_window,
            zscore_window=zscore_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        sig_a = signals["signal_a"].shift(1).fillna(0)
        sig_b = signals["signal_b"].shift(1).fillna(0)

        # Raw pair return (50/50 each leg)
        raw_return = (daily_returns[asset_a] * sig_a * 0.5 +
                      daily_returns[asset_b] * sig_b * 0.5).fillna(0)

        # Volatility scaling: scale each pair to target vol
        # Use 60-day rolling vol, shift by 1 to avoid lookahead
        rolling_vol = raw_return.rolling(60).std() * np.sqrt(252)
        rolling_vol = rolling_vol.shift(1).fillna(rolling_vol.mean())
        rolling_vol = rolling_vol.replace(0, rolling_vol.mean())

        scale = (vol_target / rolling_vol).clip(0.1, 5.0)  # cap leverage at 5x
        scaled_return = raw_return * scale

        pair_return_series[pair_name] = scaled_return

    pair_returns_df = pd.DataFrame(pair_return_series)
    n_pairs = len(pair_returns_df.columns)

    # Equal weight across pairs
    strategy_returns = pair_returns_df.sum(axis=1) / n_pairs

    first_valid = pair_returns_df.replace(0, np.nan).dropna(how="all").index[0]
    strategy_returns = strategy_returns.loc[first_valid:]

    equity_curve = (1 + strategy_returns).cumprod()

    all_tickers = list(set([t for pair in pairs for t in pair
                            if t in prices_df.columns]))
    bh_returns = daily_returns[all_tickers].mean(axis=1).loc[first_valid:]
    bh_equity = (1 + bh_returns).cumprod()

    ann = 252

    def cagr(eq):
        years = len(eq) / ann
        return (eq.iloc[-1] ** (1 / years)) - 1

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(ann)

    def max_dd(eq):
        return ((eq - eq.cummax()) / eq.cummax()).min()

    def calmar(eq):
        return cagr(eq) / abs(max_dd(eq))

    summary = pd.DataFrame({
        "Pairs Strategy": [
            round(cagr(equity_curve) * 100, 2),
            round(sharpe(strategy_returns), 2),
            round(max_dd(equity_curve) * 100, 2),
            round(strategy_returns.std() * np.sqrt(ann) * 100, 2),
            round(calmar(equity_curve), 2),
        ],
        "Buy-and-Hold": [
            round(cagr(bh_equity) * 100, 2),
            round(sharpe(bh_returns), 2),
            round(max_dd(bh_equity) * 100, 2),
            round(bh_returns.std() * np.sqrt(ann) * 100, 2),
            round(calmar(bh_equity), 2),
        ]
    }, index=["CAGR (%)", "Sharpe", "Max Drawdown (%)", "Ann. Vol (%)", "Calmar"])

    return {
        "equity_curve": equity_curve,
        "bh_equity": bh_equity,
        "strategy_returns": strategy_returns,
        "pair_returns_df": pair_returns_df,
        "summary": summary,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from signals.mean_reversion.pairs_trading import PAIRS
    from data.loaders.equity import get_prices

    TICKERS = list(set([t for pair in PAIRS for t in pair]))
    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")

    print("\nRunning pairs backtest...")
    results = run_pairs_backtest(prices_df)

    print("\n=== Performance Summary ===")
    print(results["summary"])

    print("\n=== Per-Pair Contribution (ann. vol) ===")
    pair_vols = results["pair_returns_df"].std() * np.sqrt(252) * 100
    print(pair_vols.round(2))

    eq = results["equity_curve"]
    bh = results["bh_equity"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("QuantAlpha — Pairs Trading (Vol-Scaled) vs Buy-and-Hold",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(eq.index, eq.values, label="Pairs Strategy (vol-scaled)",
             color="#4CAF50", linewidth=1.8)
    ax1.plot(bh.index, bh.values, label="Buy-and-Hold",
             color="#FF9800", linewidth=1.8, linestyle="--")
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Portfolio Value (start = 1.0)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.annotate(f"{eq.iloc[-1]:.2f}x", xy=(eq.index[-1], eq.iloc[-1]),
                 fontsize=9, color="#4CAF50", fontweight="bold")
    ax1.annotate(f"{bh.iloc[-1]:.2f}x", xy=(bh.index[-1], bh.iloc[-1]),
                 fontsize=9, color="#FF9800", fontweight="bold")

    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max

    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(
        r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\pairs_backtest.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()