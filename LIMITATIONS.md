# QuantAlpha — Known Limitations & Honest Results

This document is an honest assessment of the current strategies,
their limitations, and what would be needed to make them production-ready.
Transparency about limitations is a core principle of rigorous quant research.

---

## Strategy 1 — Cross-Sectional Momentum

### Results
| Metric | Momentum L/S | Equal-Weight B&H |
|---|---|---|
| CAGR | 9.48% | 20.44% |
| Sharpe | 0.57 | 1.20 |
| Max Drawdown | -25.94% | -28.20% |

### Known Limitations

**1. Benchmark bias**
The equal-weight benchmark includes high-momentum individual stocks
(NVDA, META, AAPL) which compounded at extraordinary rates from 2015–2026.
A fairer benchmark would be SPY alone (~12% CAGR over the same period),
against which the momentum strategy is more competitive.

**2. No transaction costs**
Every monthly rebalance incurs bid-ask spreads and market impact.
For a 56-asset universe rebalancing monthly, realistic transaction costs
would reduce CAGR by approximately 1–2% per year.

**3. Universe survivorship bias**
The universe was selected in 2026 using tickers that are currently liquid
and well-known. Some of these tickers did not exist or were illiquid in
2015, which introduces mild survivorship bias.

**4. Small universe**
Cross-sectional momentum works best with 50–500 assets. Our 56-asset
universe is at the lower bound. A larger universe (e.g. Russell 1000
constituents) would provide a more robust signal.

**5. Single signal**
The strategy uses only price momentum. Adding fundamental signals
(earnings momentum, analyst revisions) would improve the Sharpe ratio.

### What would improve it
- Expand universe to Russell 1000 or S&P 500 constituents
- Add transaction cost model (bid-ask + market impact)
- Combine with mean reversion signal for regime diversification
- Add factor exposure controls (market beta, sector neutrality)

---

## Strategy 2 — Pairs Trading (Mean Reversion)

### Results
| Metric | Pairs Strategy | Buy-and-Hold |
|---|---|---|
| CAGR | -0.53% | 21.04% |
| Sharpe | -0.10 | 1.11 |
| Max Drawdown | -12.93% | -29.15% |

### Why the strategy loses money

**1. Look-ahead bias in pair selection**
The cointegration test (Engle-Granger) was run on the full 2015–2026
dataset before backtesting. This means we selected pairs that appeared
cointegrated in hindsight. In a production system, pair selection must
be re-run on a rolling out-of-sample window (e.g. select pairs on years
1–3, trade on year 4, re-select on years 2–4, trade on year 5, etc.).

**2. Spurious cointegration**
Several selected pairs (SLV/AMD, GLD/GOOG, EWJ/GOOG) are statistically
cointegrated over the full period but have no stable economic relationship.
The spread drifts structurally rather than reverting, generating losses.
True cointegration requires both statistical significance AND a fundamental
economic rationale for mean reversion.

**3. Strategy decay**
Academic research (Gatev, Goetzmann & Rouwenhorst, 2006) documented that
pairs trading on US equities was profitable in the 1960s–1990s. Since
approximately 2002, as the strategy became widely known and adopted by
quantitative funds, excess returns have largely disappeared. The strategy
now works primarily in:
- Cryptocurrency markets (less efficient)
- Futures and FX (structural carry relationships)
- Intraday data (microstructure mean reversion)
- ETF arbitrage (ETF vs. underlying basket)

**4. High time-in-market**
With ~80% time in market across all pairs, the strategy has high gross
exposure relative to its net P&L. This amplifies transaction costs and
financing costs in a real portfolio.

### What would improve it
- Rolling out-of-sample cointegration selection (no look-ahead bias)
- Restrict pairs to within same sector/asset class with clear economic logic
- Add stop-loss per pair (exit if z-score exceeds 3.5 — spread is breaking down)
- Use intraday data for ETF arbitrage pairs
- Test on futures or FX where carry provides a fundamental anchor

---

## General Limitations

**No execution model**
All backtests assume trades execute at the closing price on the signal day.
In reality, signals are generated at close and trades execute at the next
open, with bid-ask spread and market impact. This is partially addressed
by the 1-day signal lag in the backtest engine, but execution costs are
not modeled.

**No leverage constraints**
The volatility-scaled pairs strategy can apply up to 5x leverage per pair.
Real funds face margin constraints, prime broker limits, and regulatory
capital requirements that would reduce leverage and returns.

**Single market regime**
The backtest period (2015–2026) was dominated by a US equity bull market
with one major crash (COVID-19, March 2020) and one correction (2022).
Strategy performance in prolonged bear markets or high-inflation regimes
(e.g. 1970s, 2000–2002) is untested.

---

## Interview Talking Points

These limitations are features, not bugs, when discussing this project
in MFE interviews or quant job applications:

- "I identified and documented look-ahead bias before it could mislead"
- "I understand why pairs trading decayed as a strategy post-2002"
- "The momentum strategy's drawdown control is its real value, not raw CAGR"
- "A negative result you understand is more valuable than a positive result
   you don't"

---

*Last updated: June 2026*
*Author: Diego Mella Valerio*