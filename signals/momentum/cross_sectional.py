import pandas as pd
import numpy as np


def compute_momentum(prices_df, lookback=252, skip=21):
    """Compute cross-sectional momentum scores for each asset.

    Uses the standard 12-1 momentum definition:
    - lookback: ~252 trading days (12 months)
    - skip: ~21 trading days (1 month) — skips last month to avoid
      short-term reversal contaminating the signal

    Args:
        prices_df: DataFrame of closing prices, one column per asset
        lookback:  Number of trading days to look back (default 252 = 1 year)
        skip:      Number of recent days to skip (default 21 = 1 month)

    Returns:
        DataFrame with columns:
            raw_return   — actual 12-1 return for each asset
            rank         — rank from 1 (worst) to N (best)
            score        — normalized score between 0 and 1
            signal       — long (+1), neutral (0), or short (-1)
    """
    # Drop assets with missing prices at either endpoint
    recent = prices_df.iloc[-skip]
    past = prices_df.iloc[-(lookback + skip)]
    valid = recent.notna() & past.notna()
    recent = recent[valid]
    past = past[valid]
    raw_return = (recent - past) / past

    # Rank assets (1 = worst momentum, N = best momentum)
    rank = raw_return.rank()
    n = len(rank)

    # Normalize to 0-1 score
    score = (rank - 1) / (n - 1)

    # Signal: top third = long, bottom third = short, middle = neutral
    def assign_signal(s):
        if s >= 2 / 3:
            return 1    # Long — strong momentum
        elif s <= 1 / 3:
            return -1   # Short — weak momentum
        else:
            return 0    # Neutral

    signal = score.apply(assign_signal)

    # Package results
    result = pd.DataFrame({
        "raw_return": raw_return.round(4),
        "rank": rank.astype("Int64"),
        "score": score.round(4),
        "signal": signal,
    }).sort_values("rank", ascending=False)

    return result


def compute_momentum_weights(signal_df):
    """Convert signals into portfolio weights (long/short, equal weighted).

    Long positions sum to +1, short positions sum to -1.
    Net exposure depends on how many longs vs shorts.

    Args:
        signal_df: Output of compute_momentum()

    Returns:
        Series of portfolio weights per asset
    """
    longs = (signal_df["signal"] == 1).sum()
    shorts = (signal_df["signal"] == -1).sum()

    weights = {}
    for asset, row in signal_df.iterrows():
        if row["signal"] == 1 and longs > 0:
            weights[asset] = 1 / longs        # Equal weight among longs
        elif row["signal"] == -1 and shorts > 0:
            weights[asset] = -1 / shorts      # Equal weight among shorts
        else:
            weights[asset] = 0.0

    return pd.Series(weights)


# --- Quick test when running this file directly ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")
    from data.loaders.equity import get_prices

    TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM"]
    prices_df = get_prices(TICKERS, start="2022-01-01")

    print("=== Cross-Sectional Momentum Signal ===\n")
    signal_df = compute_momentum(prices_df)
    print(signal_df)

    print("\n=== Portfolio Weights ===\n")
    weights = compute_momentum_weights(signal_df)
    print(weights)
    print(f"\nNet exposure: {weights.sum():.2f}")
    print(f"Gross exposure: {weights.abs().sum():.2f}")