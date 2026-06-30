import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")


def get_fama_french_factors(start="2015-01-01"):
    """Download Fama-French 5 factors + Momentum from Kenneth French's library.

    The factors are:
        Mkt-RF : Market excess return (market beta exposure)
        SMB    : Small Minus Big (size factor)
        HML    : High Minus Low (value factor)
        RMW    : Robust Minus Weak (profitability factor)
        CMA    : Conservative Minus Aggressive (investment factor)
        Mom    : Momentum factor (Carhart)
        RF     : Risk-free rate

    All returned as daily decimal returns (e.g. 0.01 = 1%).

    Returns:
        DataFrame of daily factor returns
    """
    import pandas_datareader.data as web

    # 5-factor model (daily)
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                         "famafrench", start=start)[0]
    # Momentum factor (daily)
    mom = web.DataReader("F-F_Momentum_Factor_daily",
                         "famafrench", start=start)[0]

    # French library returns percentages — convert to decimals
    ff5 = ff5 / 100.0
    mom = mom / 100.0

    # Clean column name for momentum (it has whitespace)
    mom.columns = ["Mom"]

    # Combine
    factors = ff5.join(mom, how="inner")

    # French data uses a PeriodIndex — convert to timestamps
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    else:
        factors.index = pd.to_datetime(factors.index)

    print(f"  Loaded {len(factors)} days of factor data")
    print(f"  Factors: {list(factors.columns)}")
    return factors


def run_factor_regression(strategy_returns, factors):
    """Regress strategy returns on Fama-French factors via OLS.

    Model:
        R_strategy - RF = alpha + b1*(Mkt-RF) + b2*SMB + b3*HML
                          + b4*RMW + b5*CMA + b6*Mom + epsilon

    The alpha is the annualized return NOT explained by any factor —
    this is the strategy's true edge. The betas show factor exposures.

    Args:
        strategy_returns: Series of daily strategy returns
        factors:          DataFrame of factor returns (from get_fama_french_factors)

    Returns:
        dict with regression results, t-stats, and interpretation
    """
    import statsmodels.api as sm

    # Align dates
    common = strategy_returns.index.intersection(factors.index)
    y = strategy_returns.loc[common]
    X = factors.loc[common]

    rf = X["RF"]
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    X_factors = X[factor_cols]

    # Excess strategy return (subtract risk-free rate)
    y_excess = y - rf

    # Add constant for alpha
    X_reg = sm.add_constant(X_factors)

    # Run OLS
    model = sm.OLS(y_excess, X_reg, missing="drop").fit()

    # Annualize alpha (daily alpha * 252)
    daily_alpha = model.params["const"]
    ann_alpha = daily_alpha * 252

    results = {
        "model": model,
        "ann_alpha": ann_alpha,
        "daily_alpha": daily_alpha,
        "alpha_tstat": model.tvalues["const"],
        "alpha_pvalue": model.pvalues["const"],
        "betas": model.params[factor_cols],
        "tstats": model.tvalues[factor_cols],
        "pvalues": model.pvalues[factor_cols],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }
    return results


def print_factor_report(results, strategy_name="Strategy"):
    """Print a formatted factor regression report.

    Args:
        results:       Output of run_factor_regression()
        strategy_name: Label for the strategy
    """
    print(f"\n{'='*60}")
    print(f"  FACTOR REGRESSION REPORT — {strategy_name}")
    print(f"{'='*60}")

    # Alpha significance
    if results["alpha_pvalue"] < 0.01:
        sig = "*** (highly significant)"
    elif results["alpha_pvalue"] < 0.05:
        sig = "** (significant)"
    elif results["alpha_pvalue"] < 0.10:
        sig = "* (weak)"
    else:
        sig = "(not significant)"

    print(f"\n  ANNUALIZED ALPHA: {results['ann_alpha']*100:+.2f}%  {sig}")
    print(f"  Alpha t-stat: {results['alpha_tstat']:.2f}  "
          f"(|t| > 2 means statistically significant)")

    print(f"\n  FACTOR EXPOSURES (betas):")
    print(f"  {'Factor':<10}{'Beta':>10}{'t-stat':>10}{'Significant':>14}")
    print(f"  {'-'*44}")

    for factor in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]:
        beta = results["betas"][factor]
        tstat = results["tstats"][factor]
        pval = results["pvalues"][factor]
        sig_mark = "yes" if pval < 0.05 else "no"
        print(f"  {factor:<10}{beta:>10.3f}{tstat:>10.2f}{sig_mark:>14}")

    print(f"\n  R-squared: {results['r_squared']:.3f}  "
          f"({results['r_squared']*100:.1f}% of returns explained by factors)")
    print(f"  Adj R-squared: {results['adj_r_squared']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    print("Running momentum backtest...")
    results_bt = run_backtest(prices_df)
    mom_returns = results_bt["strategy_returns"]

    print("\nDownloading Fama-French factors...")
    factors = get_fama_french_factors(start="2015-01-01")

    print("\nRunning factor regression...")
    results = run_factor_regression(mom_returns, factors)
    print_factor_report(results, strategy_name="Cross-Sectional Momentum")

    # ── Plot: factor betas with significance ──
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    factor_names = ["Market", "Size\n(SMB)", "Value\n(HML)",
                    "Profit.\n(RMW)", "Invest.\n(CMA)", "Momentum\n(Mom)"]
    betas = [results["betas"][f] for f in factor_cols]
    pvals = [results["pvalues"][f] for f in factor_cols]

    colors = ["#2196F3" if p < 0.05 else "#BDBDBD" for p in pvals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("QuantAlpha — Factor Exposure Analysis (Momentum Strategy)",
                 fontsize=14, fontweight="bold")

    # Left: factor betas
    bars = ax1.bar(factor_names, betas, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Factor Beta (exposure)")
    ax1.set_title("Factor Exposures\n(blue = statistically significant, gray = not)",
                  fontsize=10, color="#1F4E79")
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, beta in zip(bars, betas):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                 f"{beta:.2f}", ha="center", va=va, fontsize=9,
                 fontweight="bold")

    # Right: alpha + R-squared summary box
    ax2.axis("off")
    ann_alpha = results["ann_alpha"] * 100
    alpha_t = results["alpha_tstat"]
    r2 = results["r_squared"]

    alpha_color = "#4CAF50" if results["alpha_pvalue"] < 0.05 else "#FF9800"

    summary_text = (
        f"ANNUALIZED ALPHA\n"
        f"{ann_alpha:+.2f}%\n\n"
        f"Alpha t-stat: {alpha_t:.2f}\n"
        f"{'SIGNIFICANT' if results['alpha_pvalue'] < 0.05 else 'NOT SIGNIFICANT'}\n\n"
        f"R-squared: {r2:.3f}\n"
        f"({r2*100:.1f}% explained\nby factors)\n\n"
        f"INTERPRETATION:\n"
    )

    if results["alpha_pvalue"] < 0.05 and ann_alpha > 0:
        interp = ("The strategy generates\npositive alpha beyond\nknown factors. This is\ntrue skill.")
    elif abs(results["betas"]["Mom"]) > 0.3:
        interp = ("Returns are largely\nexplained by the\nmomentum factor — the\nstrategy replicates it.")
    else:
        interp = ("Returns are mostly\nfactor exposure, not\nunique alpha.")

    ax2.text(0.5, 0.95, summary_text + interp,
             transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", horizontalalignment="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EBF3FB",
                       edgecolor=alpha_color, linewidth=2))

    plt.tight_layout()
    plt.savefig(
        r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\factor_analysis.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()