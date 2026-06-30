# QuantAlpha

A systematic quantitative trading and portfolio management framework built in Python. Developed as the technical centerpiece for MFE program applications and a career as a quant trader / portfolio manager.

## Author
Diego Mella Valerio — MSc Financial Engineering, UAI Chile
Experience: VaR, XVA, P&L Explain @ Tanner Servicios Financieros | Murex MX.3 Test Engineer
GitHub: [dmellav-quant](https://github.com/dmellav-quant)

---

## Overview

QuantAlpha implements the full pipeline of a systematic trading operation: multi-asset data ingestion, signal generation, vectorized backtesting, risk analytics, portfolio optimization, and institutional-grade reporting. Every strategy is evaluated honestly — including documented limitations and a full factor decomposition that shows where returns actually come from.

---

## Project Structure

    QuantAlpha/
    ├── data/
    │   └── loaders/
    │       └── equity.py                ← Multi-asset price downloader (yfinance + curl_cffi)
    ├── signals/
    │   ├── momentum/
    │   │   └── cross_sectional.py       ← 12-1 cross-sectional momentum signal
    │   ├── mean_reversion/
    │   │   └── pairs_trading.py         ← Engle-Granger cointegration pairs trading
    │   └── options/
    │       └── vix_regime.py            ← VIX volatility regime filter
    ├── backtest/
    │   ├── vectorized.py                ← Vectorized backtest engine (momentum)
    │   └── pairs_backtest.py            ← Vol-scaled pairs backtest engine
    ├── risk/
    │   ├── metrics.py                   ← VaR, CVaR, Calmar, rolling risk metrics
    │   └── factors.py                   ← Fama-French 5-factor + momentum regression
    ├── notebooks/
    │   ├── 01_momentum_backtest.py      ← Momentum backtest runner
    │   ├── tearsheet.py                 ← 6-page institutional tearsheet (PDF)
    │   ├── QuantAlpha_Tearsheet.pdf     ← Generated tearsheet
    │   └── *.png                        ← Strategy & risk charts
    ├── documentation/
    │   ├── QuantAlpha_Summary.docx          ← Project summary + metrics glossary
    │   ├── QuantAlpha_Factor_Analysis.docx  ← Factor decomposition writeup
    │   └── QuantAlpha_Tearsheet_Analysis.docx ← Full tearsheet guide (with charts)
    ├── LIMITATIONS.md                   ← Honest assessment of every strategy
    └── README.md

---

## Strategies Implemented

### 1 — Cross-Sectional Momentum
12-1 momentum (12-month return, skip last month) across a 56-asset multi-asset universe spanning US/international equities, fixed income, commodities, real estate and volatility. Top third long, bottom third short, monthly rebalance.

| Metric | Momentum L/S | Equal-Weight B&H |
|---|---|---|
| CAGR | 9.08% | 20.27% |
| Sharpe | 0.55 | 1.19 |
| Max Drawdown | -25.94% | -28.20% |
| Calmar | 0.35 | 0.72 |

![Momentum Backtest](notebooks/momentum_backtest.png)

### 2 — Pairs Trading (Mean Reversion)
Engle-Granger cointegration test across all asset pairs; the top 10 statistically cointegrated pairs are traded via rolling-OLS hedge ratio and z-score entry/exit, with volatility-scaled position sizing. The strategy is market-neutral with very low volatility — and its underperformance is honestly attributed to look-ahead bias and post-2002 strategy decay (see LIMITATIONS.md).

### 3 — VIX Regime Filter (Options Signal)
Uses VIX implied volatility to classify the market into risk-on / elevated / risk-off regimes. Key finding: the momentum strategy earns a **Sharpe of 1.53 when VIX < 20** but loses money when VIX is elevated — proving the alpha is regime-dependent.

---

## Risk & Attribution

### Risk Module (`risk/metrics.py`)
Historical, parametric, and Cornish-Fisher VaR; CVaR (Expected Shortfall); Calmar; rolling Sharpe / volatility / VaR / drawdown. Directly informed by professional VaR-engine experience at Tanner.

### Factor Analysis (`risk/factors.py`)
Fama-French 5-factor + Carhart momentum regression on strategy returns:

| Factor | Beta | t-stat |
|---|---|---|
| Momentum | +0.68 | 46.1 |
| Market | +0.46 | 33.5 |
| Size / Value / Profit. / Invest. | small (-0.07 to -0.09) | significant |

**Alpha: -0.21% (not significant) · R² = 0.61.** The strategy is statistically a momentum-factor replica with unintended net-long market exposure and no true alpha — a rigorous, honest performance attribution.

---

## Institutional Tearsheet

A 6-page PDF (`notebooks/QuantAlpha_Tearsheet.pdf`) combining equity curves, rolling risk metrics, portfolio optimization (risk parity + mean-variance), full risk report, VIX regime analysis, and factor exposures. A companion guide in `documentation/` explains every chart and metric with embedded visuals.

---

## Key Technical Concepts Demonstrated

Cross-sectional ranking · 12-1 momentum · Engle-Granger cointegration · rolling OLS hedge ratios · z-score mean reversion · volatility targeting · vectorized backtesting · Historical/Parametric/Cornish-Fisher VaR · CVaR / Expected Shortfall · Calmar & rolling metrics · risk parity · mean-variance optimization · Fama-French factor attribution · VIX regime filtering.

---

## Roadmap

- [x] Multi-asset data pipeline (yfinance + curl_cffi)
- [x] Cross-sectional momentum signal + vectorized backtest
- [x] Pairs trading (Engle-Granger cointegration) + vol-scaled backtest
- [x] VIX regime filter (options signal)
- [x] Risk module (VaR, CVaR, Calmar, rolling metrics)
- [x] Fama-French 5-factor + momentum attribution
- [x] 6-page institutional tearsheet PDF
- [x] Full documentation + honest limitations
- [ ] Rolling out-of-sample cointegration (fix look-ahead bias)
- [ ] Transaction cost & slippage model
- [ ] Market-beta hedge overlay (neutralize the +0.46 market exposure)
- [ ] Combined multi-signal portfolio (momentum + mean reversion + regime)

---

## Dependencies

    yfinance · curl_cffi · pandas · numpy · matplotlib
    statsmodels · scipy · pandas-datareader

Install:

    "C:\Users\Asus\AppData\Local\spyder-6\envs\spyder-runtime\python.exe" -m pip install yfinance curl_cffi pandas numpy matplotlib statsmodels scipy pandas-datareader

---

## Documentation

- `documentation/QuantAlpha_Summary.docx` — project summary + metrics glossary
- `documentation/QuantAlpha_Factor_Analysis.docx` — factor decomposition writeup
- `documentation/QuantAlpha_Tearsheet_Analysis.docx` — full tearsheet guide with charts
- `LIMITATIONS.md` — honest assessment of what works, what doesn't, and why

---

*QuantAlpha v1.0 · Author: Diego Mella Valerio · June 2026*
*Target: MFE applications 2026–2027 cycle (Baruch · CMU · Columbia · Imperial)*