import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")

from data.loaders.equity import get_prices
from backtest.vectorized import run_backtest

# --- Load data and run backtest ---
TICKERS = [
    # US Equities — broad market & sectors
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000 small cap
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLK",   # Technology
    "XLI",   # Industrials
    "VUG",   # Vanguard Growth ETF
    "VT",    # Vanguard Total World

    # International Equities
    "EEM",   # Emerging markets
    "EFA",   # Developed markets ex-US
    "FXI",   # China
    "EWJ",   # Japan
    "EWZ",   # Brazil
    "MELI",  # MercadoLibre — LatAm ecommerce

    # Fixed Income
    "TLT",   # 20yr US Treasury
    "IEF",   # 7-10yr US Treasury
    "HYG",   # High yield corporate bonds
    "LQD",   # Investment grade corporate bonds
    "EMB",   # Emerging market bonds

    # Commodities & Volatility
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
    "DBA",   # Agriculture
    "PDBC",  # Broad commodities
    "VIXY",  # VIX short-term futures (volatility)

    # Real Estate
    "VNQ",   # US REITs
    "REM",   # Mortgage REITs

    # Semiconductors & Hardware
    "NVDA",  # Nvidia
    "AMD",   # AMD
    "INTC",  # Intel
    "SMCI",  # Super Micro Computer
    "TSM",   # Taiwan Semiconductor
    "QCOM",  # Qualcomm
    "DELL",  # Dell Technologies
    "ANET",  # Arista Networks

    # Mega-cap Tech
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOG",  # Alphabet Class C
    "GOOGL", # Alphabet Class A
    "AMZN",  # Amazon
    "META",  # Meta
    "NFLX",  # Netflix
    "TSLA",  # Tesla

    # Cybersecurity & Software
    "PANW",  # Palo Alto Networks
    "GTLB",  # GitLab

    # Consumer & Retail
    "NKE",   # Nike
    "CMG",   # Chipotle
    "WMT",   # Walmart
    "MCD",   # McDonald's
    "SPOT",  # Spotify

    # Gaming
    "TTWO",  # Take-Two Interactive

    # Airlines
    "AAL",   # American Airlines

    # Financials
    "GS",    # Goldman Sachs
    "BAC",   # Bank of America
]

prices_df = get_prices(TICKERS, start="2015-01-01")
results = run_backtest(prices_df)

eq = results["equity_curve"]
ew = results["ew_equity"]
summary = results["summary"]

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("QuantAlpha — Cross-Sectional Momentum vs Equal-Weight",
             fontsize=14, fontweight="bold", y=0.98)

# Top panel: equity curves
ax1 = axes[0]
ax1.plot(eq.index, eq.values, label="Momentum (L/S)", color="#2196F3", linewidth=1.8)
ax1.plot(ew.index, ew.values, label="Equal-Weight B&H", color="#FF9800",
         linewidth=1.8, linestyle="--")
ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
ax1.set_ylabel("Portfolio Value (start = 1.0)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Annotate final values
ax1.annotate(f"{eq.iloc[-1]:.2f}x", xy=(eq.index[-1], eq.iloc[-1]),
             fontsize=9, color="#2196F3", fontweight="bold")
ax1.annotate(f"{ew.iloc[-1]:.2f}x", xy=(ew.index[-1], ew.iloc[-1]),
             fontsize=9, color="#FF9800", fontweight="bold")

# Bottom panel: drawdown
roll_max = eq.cummax()
drawdown = (eq - roll_max) / roll_max

ax2 = axes[1]
ax2.fill_between(drawdown.index, drawdown.values, 0,
                 color="#F44336", alpha=0.4, label="Momentum Drawdown")
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\momentum_backtest.png",
            dpi=150, bbox_inches="tight")
plt.show()

# --- Print summary ---
print("\n=== Performance Summary ===")
print(summary)