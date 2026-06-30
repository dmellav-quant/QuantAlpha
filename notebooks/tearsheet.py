import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha")

from data.loaders.equity import get_prices
from backtest.vectorized import run_backtest
from risk.metrics import compute_var, compute_cvar, compute_full_risk_report
from signals.options.vix_regime import (
    get_vix, apply_vix_filter, regime_summary, classify_regime
)
from risk.factors import (
    get_fama_french_factors, run_factor_regression
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = r"C:\Users\Asus\Documents\Diego Mella Valerio\Projects\QuantAlpha\notebooks\QuantAlpha_Tearsheet.pdf"

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

OPTIM_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM", "EFA",
                 "HYG", "VNQ", "USO", "IEF"]

COLORS = {
    "momentum": "#2196F3", "benchmark": "#FF9800", "risk": "#F44336",
    "risk_dark": "#B71C1C", "green": "#4CAF50", "gray": "#9E9E9E",
    "purple": "#9C27B0",
}
STYLE = {"bg": "#FAFAFA", "title_bg": "#1F4E79", "title_fg": "white"}


# ── HELPERS ──────────────────────────────────────────────────────────────────
def cagr(eq):
    return (eq.iloc[-1] ** (252 / len(eq))) - 1
def sharpe(r):
    return (r.mean() / r.std()) * np.sqrt(252)
def max_dd(eq):
    return ((eq - eq.cummax()) / eq.cummax()).min()
def calmar(eq):
    return cagr(eq) / abs(max_dd(eq))

def banner(fig, title, subtitle=""):
    ax = fig.add_axes([0, 0.91, 1, 0.09])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 facecolor=STYLE["title_bg"], edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.5, 0.63 if subtitle else 0.5, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color="white", transform=ax.transAxes, zorder=5)
    if subtitle:
        ax.text(0.5, 0.22, subtitle, ha="center", va="center", fontsize=9,
                color="#B3D9F7", transform=ax.transAxes, zorder=5)

def mini_table(ax, df, title=""):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=STYLE["title_bg"], pad=8)
    colors_cells = []
    for i in range(len(df.index)):
        if str(df.index[i]).startswith("\u2500\u2500"):
            colors_cells.append(["#D6E4F0"] * len(df.columns))
        else:
            colors_cells.append(["#F5F5F5" if i % 2 == 0 else "white"] * len(df.columns))
    tbl = ax.table(cellText=df.values.tolist(), rowLabels=df.index.tolist(),
                   colLabels=df.columns.tolist(), cellLoc="center", rowLoc="left",
                   loc="center", cellColours=colors_cells)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(STYLE["title_bg"])
            cell.set_text_props(color="white", fontweight="bold")

def compute_risk_parity_weights(prices_df, window=252):
    returns = prices_df.pct_change().dropna()
    vols = returns.tail(window).std() * np.sqrt(252)
    inv_vol = 1 / vols
    return (inv_vol / inv_vol.sum()).sort_values(ascending=False)

def compute_mv_weights(prices_df, window=252, risk_free=0.04):
    from scipy.optimize import minimize
    returns = prices_df.pct_change().dropna().tail(window)
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(mu)
    def neg_sharpe(w):
        return -(w @ mu - risk_free) / np.sqrt(w @ sigma @ w)
    result = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                      bounds=[(0, 0.20)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                      options={"maxiter": 1000})
    weights = pd.Series(result.x, index=prices_df.columns)
    return weights[weights > 0.001].sort_values(ascending=False)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def build_tearsheet():
    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")

    print("Running momentum backtest...")
    results = run_backtest(prices_df)
    mom_returns = results["strategy_returns"]
    mom_eq      = results["equity_curve"]
    ew_ret      = prices_df.pct_change().mean(axis=1).loc[mom_returns.index]
    ew_eq       = (1 + ew_ret).cumprod()

    W = 252
    roll_sharpe = (mom_returns.rolling(W).mean() / mom_returns.rolling(W).std()) * np.sqrt(252)
    roll_vol  = mom_returns.rolling(W).std() * np.sqrt(252) * 100
    roll_var  = mom_returns.rolling(W).apply(lambda x: np.percentile(x, 5) * 100, raw=True)
    roll_cvar = mom_returns.rolling(W).apply(lambda x: x[x <= np.percentile(x, 5)].mean() * 100, raw=True)
    equity   = (1 + mom_returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax() * 100

    report = compute_full_risk_report(mom_returns, name="Momentum L/S")

    print("Computing portfolio weights...")
    optim_prices = get_prices(OPTIM_TICKERS, start="2020-01-01")
    rp_weights = compute_risk_parity_weights(optim_prices)
    try:
        mv_weights = compute_mv_weights(optim_prices)
    except Exception:
        mv_weights = rp_weights

    print("Downloading VIX...")
    vix = get_vix(start="2015-01-01")
    vix_df = apply_vix_filter(mom_returns, vix)
    vix_summary = regime_summary(vix_df)

    print("Downloading Fama-French factors...")
    factors = get_fama_french_factors(start="2015-01-01")
    factor_results = run_factor_regression(mom_returns, factors)

    v95  = compute_var(mom_returns, 0.95, "historical") * 100
    cv95 = compute_cvar(mom_returns, 0.95) * 100
    v99  = compute_var(mom_returns, 0.99, "historical") * 100
    cv99 = compute_cvar(mom_returns, 0.99) * 100
    m_cagr   = cagr(mom_eq) * 100
    ew_cagr  = cagr(ew_eq) * 100
    m_sharpe = sharpe(mom_returns)
    m_maxdd  = max_dd(mom_eq) * 100
    ew_maxdd = max_dd(ew_eq) * 100
    m_vol    = mom_returns.std() * np.sqrt(252) * 100
    m_skew   = skew(mom_returns.dropna())
    m_kurt   = kurtosis(mom_returns.dropna())
    m_calmar = calmar(mom_eq)

    print("Building tearsheet PDF...")
    with PdfPages(OUTPUT_PATH) as pdf:

        # ─── PAGE 1 — EQUITY CURVES + PERFORMANCE ──────────────────────────
        fig = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        banner(fig, "QuantAlpha \u2014 Systematic Trading Tearsheet",
               "Diego Mella Valerio  \u2502  MSc Financial Engineering, UAI  \u2502  June 2026")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                               top=0.88, bottom=0.08, left=0.08, right=0.95)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(mom_eq.index, mom_eq.values, label="Momentum L/S", color=COLORS["momentum"], linewidth=1.8)
        ax1.plot(ew_eq.index, ew_eq.values, label="Equal-Weight B&H", color=COLORS["benchmark"], linewidth=1.8, linestyle="--")
        ax1.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
        ax1.set_ylabel("Portfolio Value (start = 1.0)")
        ax1.set_title("Equity Curves (2015\u20132026)", fontweight="bold", color=STYLE["title_bg"])
        ax1.legend(loc="upper left", fontsize=9); ax1.grid(True, alpha=0.25)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.annotate(f"{mom_eq.iloc[-1]:.2f}x", xy=(mom_eq.index[-1], mom_eq.iloc[-1]), fontsize=8, color=COLORS["momentum"], fontweight="bold")
        ax1.annotate(f"{ew_eq.iloc[-1]:.2f}x", xy=(ew_eq.index[-1], ew_eq.iloc[-1]), fontsize=8, color=COLORS["benchmark"], fontweight="bold")
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown.index, drawdown.values, 0, color=COLORS["risk"], alpha=0.35)
        ax2.set_ylabel("Drawdown %")
        ax2.set_title("Strategy Drawdown", fontweight="bold", color=STYLE["title_bg"])
        ax2.grid(True, alpha=0.25); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax3 = fig.add_subplot(gs[2, 0])
        perf = pd.DataFrame({
            "Momentum L/S": [f"{m_cagr:.2f}%", f"{m_sharpe:.2f}", f"{m_maxdd:.2f}%", f"{m_vol:.2f}%", f"{m_calmar:.2f}"],
            "Equal-Weight": [f"{ew_cagr:.2f}%", f"{sharpe(ew_ret):.2f}", f"{ew_maxdd:.2f}%", f"{ew_ret.std()*np.sqrt(252)*100:.2f}%", f"{calmar(ew_eq):.2f}"],
        }, index=["CAGR", "Sharpe", "Max DD", "Ann. Vol", "Calmar"])
        mini_table(ax3, perf, title="Performance Summary")
        ax4 = fig.add_subplot(gs[2, 1])
        tail = pd.DataFrame({"Value": [f"{v95:.2f}%", f"{v99:.2f}%", f"{cv95:.2f}%", f"{cv99:.2f}%", f"{m_skew:.3f}", f"{m_kurt:.3f}"]},
                            index=["VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%", "Skewness", "Kurtosis"])
        mini_table(ax4, tail, title="Tail Risk (Daily)")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ─── PAGE 2 — ROLLING RISK METRICS ─────────────────────────────────
        fig2, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True, facecolor=STYLE["bg"])
        fig2.suptitle("QuantAlpha \u2014 Rolling Risk Metrics (1Y Window)", fontsize=13, fontweight="bold", color=STYLE["title_bg"], y=0.98)
        plt.subplots_adjust(hspace=0.35, top=0.93, bottom=0.07, left=0.09, right=0.95)
        axes[0].plot(roll_sharpe.index, roll_sharpe.values, color=COLORS["momentum"], linewidth=1.4)
        axes[0].axhline(0, color="gray", linewidth=0.7, linestyle=":")
        axes[0].axhline(1, color=COLORS["green"], linewidth=0.7, linestyle="--", alpha=0.6)
        axes[0].set_ylabel("Sharpe"); axes[0].set_title("Rolling 1Y Sharpe Ratio", fontsize=9, color=STYLE["title_bg"], fontweight="bold"); axes[0].grid(True, alpha=0.25)
        axes[1].plot(roll_vol.index, roll_vol.values, color=COLORS["benchmark"], linewidth=1.4)
        axes[1].set_ylabel("Vol %"); axes[1].set_title("Rolling 1Y Annualized Volatility", fontsize=9, color=STYLE["title_bg"], fontweight="bold"); axes[1].grid(True, alpha=0.25)
        axes[2].plot(roll_var.index, roll_var.values, color=COLORS["risk"], linewidth=1.4, label="VaR 95%")
        axes[2].plot(roll_cvar.index, roll_cvar.values, color=COLORS["risk_dark"], linewidth=1.4, linestyle="--", label="CVaR 95%")
        axes[2].set_ylabel("Daily %"); axes[2].set_title("Rolling 1Y VaR & CVaR (95%)", fontsize=9, color=STYLE["title_bg"], fontweight="bold")
        axes[2].legend(fontsize=8, loc="lower right"); axes[2].grid(True, alpha=0.25)
        axes[3].fill_between(drawdown.index, drawdown.values, 0, color=COLORS["risk"], alpha=0.35)
        axes[3].set_ylabel("Drawdown %"); axes[3].set_title("Drawdown", fontsize=9, color=STYLE["title_bg"], fontweight="bold")
        axes[3].set_xlabel("Date"); axes[3].grid(True, alpha=0.25); axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)

        # ─── PAGE 3 — PORTFOLIO OPTIMIZATION ───────────────────────────────
        fig3 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        banner(fig3, "QuantAlpha \u2014 Portfolio Optimization")
        gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.5, wspace=0.4, top=0.88, bottom=0.08, left=0.08, right=0.95)
        ax_rp = fig3.add_subplot(gs3[0, 0])
        rp_top = rp_weights.head(10)
        ax_rp.barh(rp_top.index[::-1], rp_top.values[::-1], color=COLORS["momentum"], alpha=0.85)
        ax_rp.set_title("Risk Parity Weights\n(top 10, inverse vol)", fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_rp.set_xlabel("Weight"); ax_rp.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}")); ax_rp.grid(True, alpha=0.25, axis="x")
        ax_mv = fig3.add_subplot(gs3[0, 1])
        mv_top = mv_weights.head(10)
        ax_mv.barh(mv_top.index[::-1], mv_top.values[::-1], color=COLORS["green"], alpha=0.85)
        ax_mv.set_title("Max Sharpe Weights\n(mean-variance, long-only)", fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_mv.set_xlabel("Weight"); ax_mv.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}")); ax_mv.grid(True, alpha=0.25, axis="x")
        ax_cmp = fig3.add_subplot(gs3[1, 0])
        cmp_df = pd.DataFrame({
            "Risk Parity": rp_weights.reindex(OPTIM_TICKERS).fillna(0).map(lambda x: f"{x:.1%}"),
            "Max Sharpe":  mv_weights.reindex(OPTIM_TICKERS).fillna(0).map(lambda x: f"{x:.1%}"),
        })
        mini_table(ax_cmp, cmp_df, title="Weight Comparison")
        ax_exp = fig3.add_subplot(gs3[1, 1]); ax_exp.axis("off")
        ax_exp.text(0.05, 0.95, (
            "RISK PARITY\nWeight inversely proportional to vol.\nLower-vol assets (bonds, gold) receive\nmore weight. Equal risk contribution.\nUsed by: Bridgewater All Weather.\n\n"
            "MAX SHARPE (MEAN-VARIANCE)\nMaximizes return per unit of risk.\nSensitive to return estimates \u2014 tends\nto concentrate in recent winners.\nKnown limitation: estimation error.\n\n"
            "KNOWN LIMITATION\nBoth optimizers use in-sample data.\nOut-of-sample performance degrades\ndue to estimation error. Production\nuse requires rolling reestimation."
        ), transform=ax_exp.transAxes, fontsize=8, verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="#EBF3FB", edgecolor=STYLE["title_bg"], alpha=0.8))
        pdf.savefig(fig3, bbox_inches="tight"); plt.close(fig3)

        # ─── PAGE 4 — FULL RISK REPORT + DISTRIBUTION ──────────────────────
        fig4 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        banner(fig4, "QuantAlpha \u2014 Full Risk Report")
        gs4 = gridspec.GridSpec(1, 2, figure=fig4, hspace=0.4, wspace=0.4, top=0.88, bottom=0.08, left=0.08, right=0.95)
        ax_rep = fig4.add_subplot(gs4[0, 0])
        mini_table(ax_rep, report, title="Complete Risk Metrics")
        ax_hist = fig4.add_subplot(gs4[0, 1])
        r = mom_returns.dropna()
        ax_hist.hist(r * 100, bins=80, color=COLORS["momentum"], alpha=0.7, density=True, label="Daily Returns")
        from scipy.stats import norm
        x = np.linspace(r.min() * 100, r.max() * 100, 200)
        ax_hist.plot(x, norm.pdf(x, r.mean() * 100, r.std() * 100), color=COLORS["benchmark"], linewidth=1.8, linestyle="--", label="Normal fit")
        ax_hist.axvline(v95, color=COLORS["risk"], linewidth=1.5, label=f"VaR 95%: {v95:.2f}%")
        ax_hist.axvline(cv95, color=COLORS["risk_dark"], linewidth=1.5, linestyle="--", label=f"CVaR 95%: {cv95:.2f}%")
        ax_hist.set_xlabel("Daily Return %"); ax_hist.set_ylabel("Density")
        ax_hist.set_title("Return Distribution vs Normal", fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_hist.legend(fontsize=7.5); ax_hist.grid(True, alpha=0.25)
        pdf.savefig(fig4, bbox_inches="tight"); plt.close(fig4)

        # ─── PAGE 5 — VIX REGIME FILTER ────────────────────────────────────
        fig5 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        banner(fig5, "QuantAlpha \u2014 VIX Volatility Regime Analysis",
               "How momentum performance depends on the volatility environment")
        gs5 = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.4, wspace=0.35,
                                top=0.86, bottom=0.08, left=0.09, right=0.95,
                                height_ratios=[1.3, 1])
        # VIX line with thresholds
        ax_v = fig5.add_subplot(gs5[0, :])
        ax_v.plot(vix_df["vix"].index, vix_df["vix"].values, color=COLORS["purple"], linewidth=1.0)
        ax_v.axhline(20, color=COLORS["green"], linewidth=1.2, linestyle="--", label="VIX = 20 (risk-on threshold)")
        ax_v.axhline(30, color=COLORS["risk"], linewidth=1.2, linestyle="--", label="VIX = 30 (risk-off threshold)")
        ax_v.set_ylabel("VIX Level")
        ax_v.set_title("VIX Index 2015\u20132026 with Regime Thresholds", fontsize=10, fontweight="bold", color=STYLE["title_bg"])
        ax_v.legend(loc="upper right", fontsize=8); ax_v.grid(True, alpha=0.25)
        ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        # Regime performance table
        ax_rt = fig5.add_subplot(gs5[1, 0])
        mini_table(ax_rt, vix_summary, title="Performance by VIX Regime")
        # Regime bar chart — Sharpe by regime
        ax_rb = fig5.add_subplot(gs5[1, 1])
        regimes = ["risk_on", "elevated", "risk_off"]
        regime_labels = ["Risk-On\n(VIX<20)", "Elevated\n(20-30)", "Risk-Off\n(VIX>30)"]
        sharpes = [float(vix_summary.loc[reg, "Sharpe"]) for reg in regimes]
        bar_colors = [COLORS["green"], COLORS["benchmark"], COLORS["risk"]]
        bars = ax_rb.bar(regime_labels, sharpes, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_rb.axhline(0, color="black", linewidth=0.8)
        ax_rb.set_ylabel("Sharpe Ratio")
        ax_rb.set_title("Momentum Sharpe by Regime", fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_rb.grid(True, alpha=0.25, axis="y")
        for bar, s in zip(bars, sharpes):
            h = bar.get_height()
            ax_rb.text(bar.get_x() + bar.get_width()/2, h, f"{s:.2f}", ha="center",
                       va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")
        pdf.savefig(fig5, bbox_inches="tight"); plt.close(fig5)

        # ─── PAGE 6 — FACTOR EXPOSURE ANALYSIS ─────────────────────────────
        fig6 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        banner(fig6, "QuantAlpha \u2014 Factor Exposure Analysis",
               "Fama-French 5-Factor + Momentum decomposition of strategy returns")
        gs6 = gridspec.GridSpec(2, 2, figure=fig6, hspace=0.45, wspace=0.35,
                                top=0.86, bottom=0.08, left=0.09, right=0.95,
                                height_ratios=[1.4, 1])
        # Factor betas bar chart
        ax_fb = fig6.add_subplot(gs6[0, :])
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
        factor_names = ["Market", "Size (SMB)", "Value (HML)", "Profit. (RMW)", "Invest. (CMA)", "Momentum (Mom)"]
        betas = [factor_results["betas"][f] for f in factor_cols]
        pvals = [factor_results["pvalues"][f] for f in factor_cols]
        bcolors = [COLORS["momentum"] if p < 0.05 else COLORS["gray"] for p in pvals]
        bars = ax_fb.bar(factor_names, betas, color=bcolors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_fb.axhline(0, color="black", linewidth=0.8)
        ax_fb.set_ylabel("Factor Beta (exposure)")
        ax_fb.set_title("Factor Exposures (blue = significant, gray = not)", fontsize=10, fontweight="bold", color=STYLE["title_bg"])
        ax_fb.grid(True, alpha=0.25, axis="y")
        for bar, b in zip(bars, betas):
            h = bar.get_height()
            ax_fb.text(bar.get_x() + bar.get_width()/2, h, f"{b:.2f}", ha="center",
                       va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")
        # Factor table
        ax_ft = fig6.add_subplot(gs6[1, 0])
        ftable = pd.DataFrame({
            "Beta": [f"{factor_results['betas'][f]:+.3f}" for f in factor_cols],
            "t-stat": [f"{factor_results['tstats'][f]:.2f}" for f in factor_cols],
        }, index=factor_names)
        mini_table(ax_ft, ftable, title="Factor Loadings")
        # Alpha summary box
        ax_as = fig6.add_subplot(gs6[1, 1]); ax_as.axis("off")
        ann_alpha = factor_results["ann_alpha"] * 100
        alpha_t = factor_results["alpha_tstat"]
        r2 = factor_results["r_squared"]
        acolor = COLORS["green"] if factor_results["alpha_pvalue"] < 0.05 else COLORS["benchmark"]
        ax_as.text(0.5, 0.95, (
            f"ANNUALIZED ALPHA\n{ann_alpha:+.2f}%\n(t-stat {alpha_t:.2f})\n"
            f"{'SIGNIFICANT' if factor_results['alpha_pvalue']<0.05 else 'NOT SIGNIFICANT'}\n\n"
            f"R-squared: {r2:.3f}\n{r2*100:.0f}% of returns explained\nby the six factors\n\n"
            f"CONCLUSION\nThe strategy is a near-\nreplica of the momentum\nfactor (\u03B2=0.68) with\nunintended market exposure\n(\u03B2=0.46). No true alpha."
        ), transform=ax_as.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="center",
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF3FB", edgecolor=acolor, linewidth=2))
        pdf.savefig(fig6, bbox_inches="tight"); plt.close(fig6)

    print(f"\nTearsheet saved to:\n{OUTPUT_PATH}")
    print("Done \u2014 6 pages generated.")


if __name__ == "__main__":
    build_tearsheet()