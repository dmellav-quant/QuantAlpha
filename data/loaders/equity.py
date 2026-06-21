import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")

def get_prices(tickers, start="2022-01-01"):
    """Download closing prices for multiple tickers at once.
    
    Args:
        tickers: List of ticker symbols e.g. ['SPY', 'QQQ']
        start: Start date string 'YYYY-MM-DD'
    
    Returns:
        DataFrame of closing prices, one column per ticker
    """
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    
    # Extract Close column for each ticker
    if len(tickers) == 1:
        return raw[["Close"]].rename(columns={"Close": tickers[0]})
    
    prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    return prices.dropna(how="all")


# --- Main ---
TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM"]

print("Downloading prices...")
prices_df = get_prices(TICKERS, start="2022-01-01")
returns_df = prices_df.pct_change().dropna()

print("\n=== Last 5 rows of prices ===")
print(prices_df.tail())
print("\n=== Last 5 rows of returns ===")
print(returns_df.tail())
print("\n=== Basic stats ===")
print(returns_df.describe().round(4))