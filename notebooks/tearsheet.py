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

# ── CONFIG ──────────────────────────────────────────────────────────────────
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

COLORS = {
    "momentum": "#2196F3",
    "benchmark": "#FF9800",
    "risk":      "#F44336",
    "risk_dark": "#B71C1C",
    "green":     "#4CAF50",
    "gray":      "#9E9E9E",
}

STYLE = {
    "bg":       "#FAFAFA",
    "title_bg": "#1F4E79",
    "title_fg": "white",
}


# ── HELPERS ─────────────────────────────────────────────────────────────────
def ann(r):    return 252
def cagr(eq):
    y = len(eq) / 252
    return (eq.iloc[-1] ** (1 / y)) - 1
def sharpe(r):
    return (r.mean() / r.std()) * np.sqrt(252)
def max_dd(eq):
    return ((eq - eq.cummax()) / eq.cummax()).min()
def calmar(eq):
    return cagr(eq) / abs(max_dd(eq))


def section_title(fig, text, y, fontsize=13):
    fig.text(0.5, y, text, ha="center", fontsize=fontsize,
             fontweight="bold", color=STYLE["title_bg"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6E4F0",
                       edgecolor=STYLE["title_bg"], linewidth=1.5))


def mini_table(ax, df, title=""):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=STYLE["title_bg"], pad=8)
    col_labels = df.columns.tolist()
    row_labels = df.index.tolist()
    cell_text = df.values.tolist()

    colors_header = [[STYLE["title_bg"]] * len(col_labels)]
    colors_cells = []
    for i in range(len(row_labels)):
        if df.index[i].startswith("──"):
            colors_cells.append(["#D6E4F0"] * len(col_labels))
        else:
            colors_cells.append(["#F5F5F5" if i % 2 == 0 else "white"] * len(col_labels))

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="left",
        loc="center",
        cellColours=colors_cells,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(STYLE["title_bg"])
            cell.set_text_props(color="white", fontweight="bold")


def compute_risk_parity_weights(prices_df, window=252):
    """Risk parity: weight each asset inversely proportional to its volatility."""
    returns = prices_df.pct_change().dropna()
    recent = returns.tail(window)
    vols = recent.std() * np.sqrt(252)
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.sort_values(ascending=False)


def compute_mv_weights(prices_df, window=252, risk_free=0.04):
    """Maximum Sharpe ratio (mean-variance) weights via simple optimization."""
    from scipy.optimize import minimize

    returns = prices_df.pct_change().dropna().tail(window)
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        return -(port_ret - risk_free) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.20)] * n  # max 20% per asset (long-only)
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000})

    weights = pd.Series(result.x, index=prices_df.columns)
    return weights[weights > 0.001].sort_values(ascending=False)


# ── MAIN ────────────────────────────────────────────────────────────────────
def build_tearsheet():
    print("Downloading prices...")
    prices_df = get_prices(TICKERS, start="2015-01-01")

    print("Running momentum backtest...")
    results = run_backtest(prices_df)
    mom_returns = results["strategy_returns"]
    mom_eq = results["equity_curve"]
    ew_eq = results["ew_equity"]
    ew_returns = results["strategy_returns"].copy()

    # Equal weight returns for benchmark
    ew_ret = prices_df.pct_change().mean(axis=1).loc[mom_returns.index]
    ew_eq2 = (1 + ew_ret).cumprod()

    # Rolling metrics
    window = 252
    roll_sharpe = (mom_returns.rolling(window).mean() /
                   mom_returns.rolling(window).std()) * np.sqrt(252)
    roll_vol = mom_returns.rolling(window).std() * np.sqrt(252) * 100
    roll_var = mom_returns.rolling(window).apply(
        lambda x: np.percentile(x, 5) * 100, raw=True)
    roll_cvar = mom_returns.rolling(window).apply(
        lambda x: x[x <= np.percentile(x, 5)].mean() * 100, raw=True)
    equity = (1 + mom_returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax() * 100

    # Risk report
    report = compute_full_risk_report(mom_returns, name="Momentum L/S")

    # Portfolio weights (use liquid ETF subset for speed)
    OPTIM_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM", "EFA",
                     "HYG", "VNQ", "USO", "IEF"]
    print("Computing portfolio weights...")
    optim_prices = get_prices(OPTIM_TICKERS, start="2020-01-01")
    rp_weights = compute_risk_parity_weights(optim_prices)

    try:
        mv_weights = compute_mv_weights(optim_prices)
    except Exception:
        mv_weights = rp_weights  # fallback

    print("Building tearsheet PDF...")
    with PdfPages(OUTPUT_PATH) as pdf:

        # ════════════════════════════════════════════════════════
        # PAGE 1 — COVER + EQUITY CURVES + PERFORMANCE TABLE
        # ════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               hspace=0.45, wspace=0.35,
                               top=0.88, bottom=0.08,
                               left=0.08, right=0.95)

        # Header banner
        fig.patch.set_facecolor(STYLE["bg"])
        ax_title = fig.add_axes([0, 0.91, 1, 0.09])
        ax_title.set_facecolor(STYLE["title_bg"])
        ax_title.axis("off")
        ax_title.text(0.5, 0.65, "QuantAlpha — Systematic Trading Tearsheet",
                      ha="center", va="center", fontsize=16,
                      fontweight="bold", color="white", transform=ax_title.transAxes)
        ax_title.text(0.5, 0.20,
                      "Diego Mella Valerio  |  MSc Financial Engineering, UAI  |  June 2026",
                      ha="center", va="center", fontsize=9,
                      color="#B3D9F7", transform=ax_title.transAxes)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(mom_eq.index, mom_eq.values,
                 label="Momentum L/S", color=COLORS["momentum"], linewidth=1.8)
        ax1.plot(ew_eq2.index, ew_eq2.values,
                 label="Equal-Weight B&H", color=COLORS["benchmark"],
                 linewidth=1.8, linestyle="--")
        ax1.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
        ax1.set_ylabel("Portfolio Value (start = 1.0)")
        ax1.set_title("Equity Curves (2015–2026)", fontweight="bold",
                      color=STYLE["title_bg"])
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.25)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.annotate(f"{mom_eq.iloc[-1]:.2f}x",
                     xy=(mom_eq.index[-1], mom_eq.iloc[-1]),
                     fontsize=8, color=COLORS["momentum"], fontweight="bold")
        ax1.annotate(f"{ew_eq2.iloc[-1]:.2f}x",
                     xy=(ew_eq2.index[-1], ew_eq2.iloc[-1]),
                     fontsize=8, color=COLORS["benchmark"], fontweight="bold")

        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color=COLORS["risk"], alpha=0.35, label="Drawdown")
        ax2.set_ylabel("Drawdown %")
        ax2.set_title("Strategy Drawdown", fontweight="bold",
                      color=STYLE["title_bg"])
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Performance summary table
        ax3 = fig.add_subplot(gs[2, 0])
        perf = pd.DataFrame({
            "Momentum L/S": [
                f"{cagr(mom_eq)*100:.2f}%",
                f"{sharpe(mom_returns):.2f}",
                f"{max_dd(mom_eq)*100:.2f}%",
                f"{mom_returns.std()*np.sqrt(252)*100:.2f}%",
                f"{calmar(mom_eq):.2f}",
            ],
            "Equal-Weight": [
                f"{cagr(ew_eq2)*100:.2f}%",
                f"{sharpe(ew_ret):.2f}",
                f"{max_dd(ew_eq2)*100:.2f}%",
                f"{ew_ret.std()*np.sqrt(252)*100:.2f}%",
                f"{calmar(ew_eq2):.2f}",
            ]
        }, index=["CAGR", "Sharpe", "Max DD", "Ann. Vol", "Calmar"])
        mini_table(ax3, perf, title="Performance Summary")

        # Tail risk table
        ax4 = fig.add_subplot(gs[2, 1])
        tail = pd.DataFrame({
            "Value": [
                f"{compute_var(mom_returns, 0.95, 'historical')*100:.2f}%",
                f"{compute_var(mom_returns, 0.99, 'historical')*100:.2f}%",
                f"{compute_cvar(mom_returns, 0.95)*100:.2f}%",
                f"{compute_cvar(mom_returns, 0.99)*100:.2f}%",
                f"{skew(mom_returns.dropna()):.3f}",
                f"{kurtosis(mom_returns.dropna()):.3f}",
            ]
        }, index=["VaR 95%", "VaR 99%", "CVaR 95%",
                  "CVaR 99%", "Skewness", "Kurtosis"])
        mini_table(ax4, tail, title="Tail Risk (Daily)")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ════════════════════════════════════════════════════════
        # PAGE 2 — ROLLING RISK METRICS
        # ════════════════════════════════════════════════════════
        fig2, axes = plt.subplots(4, 1, figsize=(11, 8.5),
                                  sharex=True, facecolor=STYLE["bg"])
        fig2.suptitle("QuantAlpha — Rolling Risk Metrics (1Y Window)",
                      fontsize=13, fontweight="bold", color=STYLE["title_bg"],
                      y=0.98)
        plt.subplots_adjust(hspace=0.35, top=0.93, bottom=0.07,
                            left=0.09, right=0.95)

        axes[0].plot(roll_sharpe.index, roll_sharpe.values,
                     color=COLORS["momentum"], linewidth=1.4)
        axes[0].axhline(0, color="gray", linewidth=0.7, linestyle=":")
        axes[0].axhline(1, color=COLORS["green"], linewidth=0.7,
                        linestyle="--", alpha=0.6)
        axes[0].set_ylabel("Rolling Sharpe")
        axes[0].set_title("Rolling 1Y Sharpe Ratio", fontsize=9,
                           color=STYLE["title_bg"], fontweight="bold")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(roll_vol.index, roll_vol.values,
                     color=COLORS["benchmark"], linewidth=1.4)
        axes[1].set_ylabel("Vol %")
        axes[1].set_title("Rolling 1Y Annualized Volatility",
                           fontsize=9, color=STYLE["title_bg"], fontweight="bold")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(roll_var.index, roll_var.values,
                     color=COLORS["risk"], linewidth=1.4, label="VaR 95%")
        axes[2].plot(roll_cvar.index, roll_cvar.values,
                     color=COLORS["risk_dark"], linewidth=1.4,
                     linestyle="--", label="CVaR 95%")
        axes[2].set_ylabel("Daily %")
        axes[2].set_title("Rolling 1Y VaR & CVaR (95%)",
                           fontsize=9, color=STYLE["title_bg"], fontweight="bold")
        axes[2].legend(fontsize=8, loc="lower right")
        axes[2].grid(True, alpha=0.25)

        axes[3].fill_between(drawdown.index, drawdown.values, 0,
                             color=COLORS["risk"], alpha=0.35)
        axes[3].set_ylabel("Drawdown %")
        axes[3].set_title("Drawdown", fontsize=9,
                           color=STYLE["title_bg"], fontweight="bold")
        axes[3].set_xlabel("Date")
        axes[3].grid(True, alpha=0.25)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # ════════════════════════════════════════════════════════
        # PAGE 3 — PORTFOLIO OPTIMIZATION
        # ════════════════════════════════════════════════════════
        fig3 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        fig3.suptitle("QuantAlpha — Portfolio Optimization",
                      fontsize=13, fontweight="bold",
                      color=STYLE["title_bg"], y=0.97)

        gs3 = gridspec.GridSpec(2, 2, figure=fig3,
                                hspace=0.5, wspace=0.4,
                                top=0.88, bottom=0.08,
                                left=0.08, right=0.95)

        # Risk parity bar chart
        ax_rp = fig3.add_subplot(gs3[0, 0])
        rp_top = rp_weights.head(10)
        bars = ax_rp.barh(rp_top.index[::-1], rp_top.values[::-1],
                          color=COLORS["momentum"], alpha=0.85)
        ax_rp.set_title("Risk Parity Weights\n(top 10, inverse vol)",
                         fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_rp.set_xlabel("Weight")
        ax_rp.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax_rp.grid(True, alpha=0.25, axis="x")

        # MV bar chart
        ax_mv = fig3.add_subplot(gs3[0, 1])
        mv_top = mv_weights.head(10)
        ax_mv.barh(mv_top.index[::-1], mv_top.values[::-1],
                   color=COLORS["green"], alpha=0.85)
        ax_mv.set_title("Max Sharpe Weights\n(mean-variance, long-only)",
                         fontsize=9, fontweight="bold", color=STYLE["title_bg"])
        ax_mv.set_xlabel("Weight")
        ax_mv.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax_mv.grid(True, alpha=0.25, axis="x")

        # Comparison table
        ax_cmp = fig3.add_subplot(gs3[1, 0])
        common = rp_weights.index.intersection(mv_weights.index)
        cmp_df = pd.DataFrame({
            "Risk Parity": rp_weights.reindex(OPTIM_TICKERS).fillna(0).map(
                lambda x: f"{x:.1%}"),
            "Max Sharpe": mv_weights.reindex(OPTIM_TICKERS).fillna(0).map(
                lambda x: f"{x:.1%}"),
        })
        mini_table(ax_cmp, cmp_df, title="Weight Comparison")

        # Explanation box
        ax_exp = fig3.add_subplot(gs3[1, 1])
        ax_exp.axis("off")
        explanation = (
            "RISK PARITY\n"
            "Weight inversely proportional to vol.\n"
            "Lower-vol assets (bonds, gold) receive\n"
            "more weight. Equal risk contribution.\n"
            "Used by: Bridgewater All Weather.\n\n"
            "MAX SHARPE (MEAN-VARIANCE)\n"
            "Maximizes return per unit of risk.\n"
            "Sensitive to return estimates — tends\n"
            "to concentrate in recent winners.\n"
            "Known limitation: estimation error.\n\n"
            "KNOWN LIMITATION\n"
            "Both optimizers use in-sample data.\n"
            "Out-of-sample performance typically\n"
            "degrades due to estimation error.\n"
            "Production use requires rolling\n"
            "reestimation and robust shrinkage."
        )
        ax_exp.text(0.05, 0.95, explanation, transform=ax_exp.transAxes,
                    fontsize=8, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#EBF3FB",
                              edgecolor=STYLE["title_bg"], alpha=0.8))

        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # ════════════════════════════════════════════════════════
        # PAGE 4 — FULL RISK REPORT + RETURN DISTRIBUTION
        # ════════════════════════════════════════════════════════
        fig4 = plt.figure(figsize=(11, 8.5), facecolor=STYLE["bg"])
        fig4.suptitle("QuantAlpha — Full Risk Report",
                      fontsize=13, fontweight="bold",
                      color=STYLE["title_bg"], y=0.97)

        gs4 = gridspec.GridSpec(1, 2, figure=fig4,
                                hspace=0.4, wspace=0.4,
                                top=0.88, bottom=0.08,
                                left=0.08, right=0.95)

        # Full risk report table
        ax_rep = fig4.add_subplot(gs4[0, 0])
        mini_table(ax_rep, report, title="Complete Risk Metrics")

        # Return distribution histogram
        ax_hist = fig4.add_subplot(gs4[0, 1])
        r = mom_returns.dropna()
        ax_hist.hist(r * 100, bins=80, color=COLORS["momentum"],
                     alpha=0.7, density=True, label="Daily Returns")

        # Overlay normal distribution
        from scipy.stats import norm
        x = np.linspace(r.min() * 100, r.max() * 100, 200)
        ax_hist.plot(x, norm.pdf(x, r.mean() * 100, r.std() * 100),
                     color=COLORS["benchmark"], linewidth=1.8,
                     linestyle="--", label="Normal fit")

        # Mark VaR and CVaR
        var95 = compute_var(r, 0.95, "historical") * 100
        cvar95 = compute_cvar(r, 0.95) * 100
        ax_hist.axvline(var95, color=COLORS["risk"], linewidth=1.5,
                        linestyle="-", label=f"VaR 95%: {var95:.2f}%")
        ax_hist.axvline(cvar95, color=COLORS["risk_dark"], linewidth=1.5,
                        linestyle="--", label=f"CVaR 95%: {cvar95:.2f}%")

        ax_hist.set_xlabel("Daily Return %")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Return Distribution vs Normal",
                           fontsize=9, fontweight="bold",
                           color=STYLE["title_bg"])
        ax_hist.legend(fontsize=7.5)
        ax_hist.grid(True, alpha=0.25)

        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    print(f"\nTearsheet saved to:\n{OUTPUT_PATH}")
    print("Done — 4 pages generated.")


if __name__ == "__main__":
    build_tearsheet()