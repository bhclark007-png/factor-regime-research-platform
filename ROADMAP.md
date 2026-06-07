# Roadmap

## Phase 1: Core Data And Brief

- Data ingestion from FRED, Yahoo Finance, Treasury, Cboe, BLS, local seed files, and optional vendor APIs.
- Factor scoring for momentum, quality, value, low volatility, and small cap.
- Daily brief generation in Markdown and JSON-backed dashboard format.
- Reproducible run history and output artifacts.

## Phase 2: Regime Research Platform

- Credit Leadership Score from HY OAS, IG OAS, CCC spreads, and loan proxies where available.
- Historical analog engine for similar macro and credit environments.
- Regime Stability Score and dynamic regime-transition monitoring.
- Better validation of regime labels and factor forecast horizons.

## Phase 3: Portfolio Tracking

- Portfolio Tracking Module.
- Compare user portfolio exposures to factor recommendations.
- Identify regime-specific risks.
- Monitor factor drift.
- Generate allocation recommendations with explicit confidence and caveats.

## Phase 4: Security Selection Research

- Research single-name indicators inside favored factors.
- Evaluate quality, momentum, value, revisions, profitability, and credit-improvement signals.
- Add security-level backtests before any portfolio recommendation workflow.
