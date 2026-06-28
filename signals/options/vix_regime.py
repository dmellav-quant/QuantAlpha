import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


def get_vix(start="2015-01-01"):
    """Download VIX index data from FRED (Federal Reserve Economic Data).

    VIX = CBOE Volatility Index. Measures implied volatility of S&P 500
    options over the next 30 days. Known as the market 'fear gauge'.

    Returns:
        Series of daily VIX levels, indexed by date
    """
    try:
        import pandas_datareader.data as web
        vix = web.DataReader("VIXCLS", "fred", start=start)
        vix = vix["VIXCLS"].dropna()
        vix.name = "VIX"
        print(f"  VIX loaded from FRED: {len(vix)} rows")
        return vix
    except Exception as e:
        print(f"  FRED failed ({e}), falling back to yfinance...")
        import yfinance as yf
        vix = yf.download("^VIX", start=start, progress=False, auto_adjust=True)
        vix = vix["Close"].squeeze()
        vix.name = "VIX"
        vix.index = pd.to_datetime(vix.index)
        if isinstance(vix.index, pd.MultiIndex):
            vix.index = vix.index.get_level_values(0)
        print(f"  VIX loaded from yfinance: {len(vix)} rows")
        return vix


def classify_regime(vix_series, low_threshold=20, high_threshold=30):
    """Classify each day into a volatility regime based on VIX level.

    Regimes:
        'risk_on'   — VIX < low_threshold   (calm markets, momentum works)
        'elevated'  — low <= VIX < high     (caution, stay flat)
        'risk_off'  — VIX >= high_threshold (stressed markets, stay flat)

    Args:
        vix_series:      Series of VIX levels
        low_threshold:   VIX below this = risk-on (default 20)
        high_threshold:  VIX above this = risk-off (default 30)

    Returns:
        Series of regime labels: 'risk_on', 'elevated', 'risk_off'
    """
    regime = pd.Series("elevated", index=vix_series.index, name="regime")
    regime[vix_series < low_threshold]   = "risk_on"
    regime[vix_series >= high_threshold] = "risk_off"
    return regime


def compute_regime_scalar(vix_series, low_threshold=20, high_threshold=30):
    """Convert VIX regime into a position size scalar (0 to 1).

    Key insight from regime analysis: the strategy loses money whenever
    VIX > 20, so the optimal approach is to only trade in calm regimes.

    Scalar:
        risk_on   → 1.0  (VIX < 20, full position)
        elevated  → 0.0  (20 <= VIX < 30, flat — strategy loses here)
        risk_off  → 0.0  (VIX >= 30, flat — crisis, stay out)

    Args:
        vix_series:      Series of VIX levels
        low_threshold:   VIX below this = full size
        high_threshold:  VIX above this = flat

    Returns:
        Series of scalars between 0 and 1
    """
    regime = classify_regime(vix_series, low_threshold, high_threshold)
    # Key change: elevated → 0.0 instead of 0.5
    scalar = regime.map({"risk_on": 1.0, "elevated": 0.0, "risk_off": 0.0})
    return scalar.rename("regime_scalar")


def apply_vix_filter(strategy_returns, vix_series,
                     low_threshold=20, high_threshold=30):
    """Apply VIX regime filter to an existing strategy's daily returns.

    Multiplies each day's return by the regime scalar (0 or 1.0).
    The scalar is shifted by 1 day to avoid lookahead bias:
    VIX on day T determines position size on day T+1.

    Args:
        strategy_returns: Series of daily strategy returns
        vix_series:       Series of daily VIX levels
        low_threshold:    VIX below this = full position (default 20)
        high_threshold:   VIX above this = flat (default 30)

    Returns:
        DataFrame with columns: original, filtered, vix, regime, scalar
    """
    common = strategy_returns.index.intersection(vix_series.index)
    ret    = strategy_returns.loc[common]
    vix    = vix_series.loc[common]

    scalar           = compute_regime_scalar(vix, low_threshold, high_threshold)
    scalar_lagged    = scalar.shift(1).fillna(1.0)
    filtered_returns = ret * scalar_lagged
    regime           = classify_regime(vix, low_threshold, high_threshold)

    return pd.DataFrame({
        "original": ret,
        "filtered": filtered_returns,
        "vix":      vix,
        "regime":   regime,
        "scalar":   scalar_lagged,
    })


def regime_summary(filtered_df):
    """Summarize time spent in each regime and performance per regime."""
    ann  = 252
    rows = []

    for regime in ["risk_on", "elevated", "risk_off"]:
        mask   = filtered_df["regime"] == regime
        subset = filtered_df.loc[mask, "original"]
        days   = mask.sum()
        pct    = days / len(filtered_df) * 100

        if len(subset) > 10:
            ann_ret = subset.mean() * ann * 100
            ann_vol = subset.std() * np.sqrt(ann) * 100
            sr      = (subset.mean() / subset.std()) * np.sqrt(ann)
        else:
            ann_ret = ann_vol = sr = 0

        rows.append({
            "Regime":      regime,
            "Days":        days,
            "% Time":      f"{pct:.1f}%",
            "Ann. Return": f"{ann_ret:.2f}%",
            "Ann. Vol":    f"{ann_vol:.2f}%",
            "Sharpe":      f"{sr:.2f}",
        })

    return pd.DataFrame(rows).set_index("Regime")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from data.loaders.equity import get_prices
    from backtest.vectorized import run_backtest

    TICKERS = [
        "SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "XLK", "XLI", "VUG", "VT",
        "EEM", "EFA", "FXI", "EWJ", "EWZ", "MELI",
        "TLT", "IEF", "HYG", "LQD", "EMB",
        "GLD", "SLV", "USO", "DBA", "PDBC", "VIXY",
        "VNQ", "REM",
        "NVDA", "AMD", "INTC", "SMCI", "TSM", "QCOM", "DELL", "ANET",
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NFLX", "TSLA",
        "PANW", "GTLB", "NKE", "CMG", "WMT", "MCD", "SPOT",
        "TTWO", "AAL", "GS", "BAC",
    ]

    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")

    print("Running base momentum backtest...")
    results     = run_backtest(prices_df)
    mom_returns = results["strategy_returns"]

    print("\nDownloading VIX...")
    vix = get_vix(start="2015-01-01")

    print("\nApplying VIX regime filter (risk-on only, VIX < 20)...")
    filtered_df = apply_vix_filter(mom_returns, vix)

    # ── Performance comparison ───────────────────────────────────────────────
    orig    = filtered_df["original"]
    filt    = filtered_df["filtered"]
    orig_eq = (1 + orig).cumprod()
    filt_eq = (1 + filt).cumprod()

    def cagr(eq):
        return (eq.iloc[-1] ** (252 / len(eq))) - 1
    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(252)
    def max_dd(eq):
        return ((eq - eq.cummax()) / eq.cummax()).min()

    print("\n=== Performance: Base vs VIX-Filtered Momentum ===")
    summary = pd.DataFrame({
        "Base Momentum": [
            f"{cagr(orig_eq)*100:.2f}%",
            f"{sharpe(orig):.2f}",
            f"{max_dd(orig_eq)*100:.2f}%",
            f"{orig.std()*np.sqrt(252)*100:.2f}%",
        ],
        "VIX-Filtered\n(risk-on only)": [
            f"{cagr(filt_eq)*100:.2f}%",
            f"{sharpe(filt):.2f}",
            f"{max_dd(filt_eq)*100:.2f}%",
            f"{filt.std()*np.sqrt(252)*100:.2f}%",
        ]
    }, index=["CAGR", "Sharpe", "Max Drawdown", "Ann. Vol"])
    print(summary)

    print("\n=== Regime Analysis ===")
    print(regime_summary(filtered_df))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("QuantAlpha — VIX Regime Filter on Momentum Strategy\n"
                 "Risk-On Only (VIX < 20): Full Size | VIX \u2265 20: Flat",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, top=0.91, bottom=0.07,
                        left=0.09, right=0.95)

    # Panel 1 — VIX with regime shading
    axes[0].plot(filtered_df["vix"].index, filtered_df["vix"].values,
                 color="#9C27B0", linewidth=1.2, label="VIX")
    axes[0].axhline(20, color="#4CAF50", linewidth=1.5,
                    linestyle="--", label="Threshold (20) — trade below, flat above")
    axes[0].fill_between(filtered_df.index, 0, 20,
                         alpha=0.15, color="#4CAF50", label="Risk-On zone (VIX < 20)")
    axes[0].fill_between(filtered_df.index, 20, filtered_df["vix"].max() + 5,
                         alpha=0.08, color="#F44336", label="Flat zone (VIX \u2265 20)")
    axes[0].set_ylabel("VIX Level")
    axes[0].set_title("VIX Index — Trade only when VIX < 20",
                       fontweight="bold", color="#1F4E79")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Equity curves
    axes[1].plot(orig_eq.index, orig_eq.values,
                 label="Base Momentum (always on)",
                 color="#2196F3", linewidth=1.8, linestyle="--", alpha=0.7)
    axes[1].plot(filt_eq.index, filt_eq.values,
                 label="VIX-Filtered (risk-on only)",
                 color="#4CAF50", linewidth=1.8)
    axes[1].axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
    axes[1].set_ylabel("Portfolio Value (start = 1.0)")
    axes[1].set_title("Equity Curves: Base vs VIX-Filtered",
                       fontweight="bold", color="#1F4E79")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate(f"{orig_eq.iloc[-1]:.2f}x",
                     xy=(orig_eq.index[-1], orig_eq.iloc[-1]),
                     fontsize=8, color="#2196F3", fontweight="bold")
    axes[1].annotate(f"{filt_eq.iloc[-1]:.2f}x",
                     xy=(filt_eq.index[-1], filt_eq.iloc[-1]),
                     fontsize=8, color="#4CAF50", fontweight="bold")

    # Panel 3 — Position size
    regime_colors = {"risk_on": "#4CAF50", "elevated": "#FF9800", "risk_off": "#F44336"}
    regime_labels = {
        "risk_on":  "Risk-On (VIX < 20) — Full size (100%)",
        "elevated": "Elevated (20 \u2264 VIX < 30) — Flat (0%)",
        "risk_off": "Risk-Off (VIX \u2265 30) — Flat (0%)",
    }
    for regime, color in regime_colors.items():
        mask = filtered_df["regime"] == regime
        axes[2].fill_between(
            filtered_df.index,
            filtered_df["scalar"].where(mask, np.nan),
            0,
            color=color, alpha=0.75,
            label=regime_labels[regime]
        )

    axes[2].set_ylabel("Position Size")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Position Size by VIX Regime",
                       fontweight="bold", color="#1F4E79")
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].set_yticks([0, 0.5, 1.0])
    axes[2].set_yticklabels(["0%\n(Flat)", "50%", "100%\n(Full)"])
    axes[2].legend(loc="upper right", fontsize=8,
                   framealpha=0.9, edgecolor="#CCCCCC")
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.savefig(
        r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\vix_regime.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()