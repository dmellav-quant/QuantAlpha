import pandas as pd
import numpy as np
import sys
from itertools import combinations

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


# ── Install statsmodels if needed:
# "C:\Users\Asus\AppData\Local\spyder-6\envs\spyder-runtime\python.exe" -m pip install statsmodels
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(prices_df, pvalue_threshold=0.05, top_n=10):
    """Test all possible pairs for cointegration using Engle-Granger test.

    Only pairs with a statistically significant cointegration relationship
    (p-value < threshold) are selected. This replaces manual pair selection
    with a data-driven approach.

    Args:
        prices_df:         DataFrame of closing prices
        pvalue_threshold:  Max p-value to consider a pair cointegrated
        top_n:             Maximum number of pairs to return (best p-values)

    Returns:
        List of (asset_A, asset_B, p_value) tuples, sorted by p-value
    """
    tickers = prices_df.columns.tolist()
    cointegrated = []

    print(f"  Testing {len(list(combinations(tickers, 2)))} pairs for cointegration...")

    for a, b in combinations(tickers, 2):
        series_a = prices_df[a].dropna()
        series_b = prices_df[b].dropna()

        # Align on common dates
        common = series_a.index.intersection(series_b.index)
        if len(common) < 252:   # need at least 1 year of data
            continue

        try:
            _, pvalue, _ = coint(series_a.loc[common], series_b.loc[common])
            if pvalue < pvalue_threshold:
                cointegrated.append((a, b, round(pvalue, 4)))
        except Exception:
            continue

    # Sort by p-value (most cointegrated first)
    cointegrated.sort(key=lambda x: x[2])
    return cointegrated[:top_n]


def compute_hedge_ratio(price_a, price_b, window=60):
    """Rolling OLS hedge ratio: how many units of B per unit of A."""
    hedge_ratio = pd.Series(index=price_a.index, dtype=float)
    for i in range(window, len(price_a)):
        y = price_a.iloc[i - window:i].values
        x = price_b.iloc[i - window:i].values
        beta = np.cov(x, y)[0, 1] / np.var(x)
        hedge_ratio.iloc[i] = beta
    return hedge_ratio


def compute_spread(price_a, price_b, hedge_ratio):
    """Compute the spread: spread = price_A - hedge_ratio * price_B."""
    return price_a - hedge_ratio * price_b


def compute_zscore(spread, window=20):
    """Normalize spread into a z-score using a rolling window."""
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / rolling_std


def compute_pair_signal(price_a, price_b,
                        hedge_window=60, zscore_window=20,
                        entry_threshold=1.5, exit_threshold=0.25):
    """Full pipeline for one pair: hedge ratio → spread → z-score → signal."""
    hedge_ratio = compute_hedge_ratio(price_a, price_b, window=hedge_window)
    spread = compute_spread(price_a, price_b, hedge_ratio)
    zscore = compute_zscore(spread, window=zscore_window)

    signal_a = pd.Series(0.0, index=price_a.index)
    signal_b = pd.Series(0.0, index=price_a.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if pd.isna(z):
            signal_a.iloc[i] = 0
            signal_b.iloc[i] = 0
            continue

        if position == 0:
            if z > entry_threshold:
                position = -1
            elif z < -entry_threshold:
                position = 1
        else:
            if abs(z) < exit_threshold:
                position = 0

        signal_a.iloc[i] = position
        signal_b.iloc[i] = -position

    return pd.DataFrame({
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "zscore": zscore,
        "signal_a": signal_a,
        "signal_b": signal_b,
    })


def compute_all_pairs(prices_df, pairs=None, **kwargs):
    """Run the pairs signal. If pairs=None, auto-selects via cointegration."""
    if pairs is None:
        print("  Auto-selecting pairs via cointegration test...")
        coint_pairs = find_cointegrated_pairs(prices_df)
        pairs = [(a, b) for a, b, _ in coint_pairs]
        print(f"  Found {len(pairs)} cointegrated pairs:")
        for a, b, pv in coint_pairs:
            print(f"    {a}/{b}  p-value: {pv}")

    results = {}
    for asset_a, asset_b in pairs:
        if asset_a not in prices_df.columns or asset_b not in prices_df.columns:
            continue
        pair_name = f"{asset_a}/{asset_b}"
        signals = compute_pair_signal(
            prices_df[asset_a], prices_df[asset_b], **kwargs
        )
        results[pair_name] = signals

    return results


if __name__ == "__main__":
    from data.loaders.equity import get_prices

    # Use a focused universe — cointegration works best within asset classes
    TICKERS = [
        "SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "XLK", "XLI",
        "GLD", "SLV", "USO",
        "TLT", "IEF", "HYG", "LQD",
        "EEM", "EFA", "EWZ", "FXI", "EWJ",
        "NVDA", "AMD", "INTC", "QCOM",
        "MSFT", "AAPL", "GOOG", "META",
        "GS", "BAC", "XLF",
    ]

    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df) * 0.9))

    print("\nFinding cointegrated pairs...")
    coint_pairs = find_cointegrated_pairs(prices_df, pvalue_threshold=0.05, top_n=10)

    print(f"\n=== Top Cointegrated Pairs ===")
    for a, b, pv in coint_pairs:
        print(f"  {a:6s} / {b:6s}  p-value: {pv:.4f}")

    print("\nComputing signals for cointegrated pairs...")
    pairs = [(a, b) for a, b, _ in coint_pairs]
    all_signals = compute_all_pairs(prices_df, pairs=pairs)

    for pair_name, signals in all_signals.items():
        trades = (signals["signal_a"].diff().abs() > 0).sum()
        time_in = (signals["signal_a"] != 0).mean()
        print(f"  {pair_name:15s}  trades: {trades:3d}  time in market: {time_in:.1%}")