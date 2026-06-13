# Roadmap

## Phase 1: Core Data And Brief

- Completed: data ingestion from FRED, Yahoo Finance, Treasury, Cboe, BLS, Kenneth French, local seed files, and optional vendor APIs.
- Completed: factor scoring for momentum, quality, value, low volatility, and small cap.
- Completed: daily brief generation in Markdown and JSON-backed dashboard format.
- Completed: reproducible run history, data-quality gates, and canonical output artifacts.

## Phase 2: Regime Research Platform

- Completed: Credit Leadership Score from HY OAS, IG OAS, CCC spreads, and loan proxies where available.
- Completed: historical analog engine for similar macro and credit environments.
- Completed: Regime Stability Score and regime-break transition monitoring.
- Completed: dashboard-only validation of factor forecast horizons against simple baselines.
- Next: deeper transition-label validation and loan-market proxy review.

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
