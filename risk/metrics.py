import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


def compute_var(returns, confidence=0.95, method="historical"):
    """Compute Value at Risk (VaR) for a return series.

    VaR answers: what is the maximum loss on X% of trading days?
    A daily VaR of -1.5% at 95% means 95% of days lose less than 1.5%.

    Args:
        returns:    Series of daily returns
        confidence: Confidence level (default 0.95 = 95%)
        method:     'historical', 'parametric', or 'cornish_fisher'

    Returns:
        float: VaR as a negative number (e.g. -0.015 = -1.5%)
    """
    if method == "historical":
        # Sort returns and take the (1-confidence) percentile
        return float(np.percentile(returns.dropna(), (1 - confidence) * 100))

    elif method == "parametric":
        # Assume normal distribution: VaR = mean - z * std
        from scipy.stats import norm
        mu = returns.mean()
        sigma = returns.std()
        z = norm.ppf(1 - confidence)
        return float(mu + z * sigma)

    elif method == "cornish_fisher":
        # Cornish-Fisher expansion: adjusts for skewness and kurtosis
        # More accurate than parametric for fat-tailed return distributions
        from scipy.stats import norm, skew, kurtosis
        mu = returns.mean()
        sigma = returns.std()
        s = skew(returns.dropna())
        k = kurtosis(returns.dropna())  # excess kurtosis
        z = norm.ppf(1 - confidence)
        # Cornish-Fisher adjusted z-score
        z_cf = (z +
                (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * k / 24 -
                (2*z**3 - 5*z) * s**2 / 36)
        return float(mu + z_cf * sigma)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'historical', 'parametric', or 'cornish_fisher'")


def compute_cvar(returns, confidence=0.95):
    """Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR answers: given that we are in the worst X% of days, what is
    the average loss? Always worse than VaR. Also called Expected Shortfall (ES).

    Args:
        returns:    Series of daily returns
        confidence: Confidence level (default 0.95 = 95%)

    Returns:
        float: CVaR as a negative number
    """
    var = compute_var(returns, confidence=confidence, method="historical")
    # Average of all returns worse than VaR
    tail = returns[returns <= var]
    return float(tail.mean())


def compute_rolling_metrics(returns, window=252):
    """Compute rolling risk metrics over a sliding window.

    Useful for seeing how risk and performance evolved over time,
    and for detecting regime changes.

    Args:
        returns: Series of daily returns
        window:  Rolling window in trading days (default 252 = 1 year)

    Returns:
        DataFrame with rolling Sharpe, Vol, VaR, CVaR, and Drawdown
    """
    ann = 252

    rolling_sharpe = (
        returns.rolling(window).mean() /
        returns.rolling(window).std()
    ) * np.sqrt(ann)

    rolling_vol = returns.rolling(window).std() * np.sqrt(ann)

    rolling_var = returns.rolling(window).apply(
        lambda x: np.percentile(x, 5), raw=True
    )

    rolling_cvar = returns.rolling(window).apply(
        lambda x: x[x <= np.percentile(x, 5)].mean(), raw=True
    )

    # Drawdown
    equity = (1 + returns).cumprod()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max

    return pd.DataFrame({
        "rolling_sharpe":  rolling_sharpe,
        "rolling_vol":     rolling_vol,
        "rolling_var_95":  rolling_var,
        "rolling_cvar_95": rolling_cvar,
        "drawdown":        drawdown,
    })


def compute_full_risk_report(returns, name="Strategy"):
    """Generate a complete risk report for a return series.

    Covers all standard metrics used in institutional risk management:
    return metrics, risk metrics, tail risk, and distribution stats.

    Args:
        returns: Series of daily returns
        name:    Label for the strategy (used in output)

    Returns:
        DataFrame with all risk metrics
    """
    from scipy.stats import skew, kurtosis

    ann = 252
    equity = (1 + returns).cumprod()

    # Return metrics
    total_return = equity.iloc[-1] - 1
    years = len(returns) / ann
    cagr = (equity.iloc[-1] ** (1 / years)) - 1
    ann_vol = returns.std() * np.sqrt(ann)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(ann)

    # Drawdown metrics
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd)

    # Tail risk
    var_95_hist = compute_var(returns, 0.95, "historical")
    var_99_hist = compute_var(returns, 0.99, "historical")
    var_95_param = compute_var(returns, 0.95, "parametric")
    var_95_cf = compute_var(returns, 0.95, "cornish_fisher")
    cvar_95 = compute_cvar(returns, 0.95)
    cvar_99 = compute_cvar(returns, 0.99)

    # Distribution stats
    s = skew(returns.dropna())
    k = kurtosis(returns.dropna())  # excess kurtosis (normal = 0)
    best_day = returns.max()
    worst_day = returns.min()
    pct_positive = (returns > 0).mean()

    metrics = {
        "── RETURN METRICS ──": "",
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "── DRAWDOWN ──": "",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar Ratio": f"{calmar:.2f}",
        "── TAIL RISK (daily) ──": "",
        "VaR 95% Historical": f"{var_95_hist:.2%}",
        "VaR 99% Historical": f"{var_99_hist:.2%}",
        "VaR 95% Parametric": f"{var_95_param:.2%}",
        "VaR 95% Cornish-Fisher": f"{var_95_cf:.2%}",
        "CVaR 95% (Exp. Shortfall)": f"{cvar_95:.2%}",
        "CVaR 99% (Exp. Shortfall)": f"{cvar_99:.2%}",
        "── DISTRIBUTION ──": "",
        "Skewness": f"{s:.3f}",
        "Excess Kurtosis": f"{k:.3f}",
        "Best Day": f"{best_day:.2%}",
        "Worst Day": f"{worst_day:.2%}",
        "% Positive Days": f"{pct_positive:.1%}",
    }

    return pd.DataFrame.from_dict(metrics, orient="index", columns=[name])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from data.loaders.equity import get_prices
    from backtest.vectorized import run_backtest

    # ── Load momentum strategy returns ──
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

    print("Running backtest...")
    results = run_backtest(prices_df)
    returns = results["strategy_returns"]

    # ── Full risk report ──
    print("\n=== Full Risk Report — Momentum Strategy ===")
    report = compute_full_risk_report(returns, name="Momentum L/S")
    print(report.to_string())

    # ── Rolling metrics chart ──
    print("\nComputing rolling metrics...")
    rolling = compute_rolling_metrics(returns, window=252)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("QuantAlpha — Rolling Risk Metrics (Momentum Strategy)",
                 fontsize=14, fontweight="bold")

    # Rolling Sharpe
    axes[0].plot(rolling.index, rolling["rolling_sharpe"],
                 color="#2196F3", linewidth=1.5)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].axhline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Rolling Sharpe (1Y)")
    axes[0].grid(True, alpha=0.3)

    # Rolling Vol
    axes[1].plot(rolling.index, rolling["rolling_vol"] * 100,
                 color="#FF9800", linewidth=1.5)
    axes[1].set_ylabel("Rolling Vol % (1Y)")
    axes[1].grid(True, alpha=0.3)

    # Rolling VaR and CVaR
    axes[2].plot(rolling.index, rolling["rolling_var_95"] * 100,
                 color="#F44336", linewidth=1.5, label="VaR 95%")
    axes[2].plot(rolling.index, rolling["rolling_cvar_95"] * 100,
                 color="#B71C1C", linewidth=1.5, linestyle="--", label="CVaR 95%")
    axes[2].set_ylabel("Daily VaR/CVaR %")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    # Drawdown
    axes[3].fill_between(rolling.index, rolling["drawdown"] * 100, 0,
                         color="#F44336", alpha=0.4)
    axes[3].set_ylabel("Drawdown %")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(
        r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\risk_metrics.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()