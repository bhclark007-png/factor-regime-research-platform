# Changelog

## 0.5.0 - Validation Framework

- Added multi-horizon walk-forward validation for 1M, 3M, and 6M factor leadership.
- Compared model-selected factor exposure against equal-weight factors and SPY.
- Added hit rate, average forward excess return, information ratio, max drawdown, turnover proxy, and confusion matrices.
- Added dashboard-only validation section while preserving the daily brief format.

## 0.4.0 - Repository Platform Scaffold

- Moved importable code under `src/factor_agent`.
- Added long-term documentation set and ADRs.
- Added historical analog engine.
- Added dynamic percentile-based regime risk monitoring.
- Added reproducible backtest metrics module.
- Updated daily brief and dashboard to include analog/risk outputs.

## 0.3.0 - Dashboard

- Added Streamlit dashboard over `output/latest/run_result.json`.
- Added data-source health, run history, model drivers, and daily brief display.

## 0.2.0 - Data Reliability

- Added source status reporting, local cache, timestamped run outputs, and latest artifact contract.
- Added Treasury, Cboe, BLS, PMI proxy, and CCC seed/update fallback paths.

## 0.1.0 - Prototype

- Initial factor-regime CLI.
- Generated factor probabilities, Credit Leadership Score, Regime Stability Score, backtest summary, and Markdown brief.
