# Changelog

## v0.6.1 - Quality And Test Hardening

- Reformatted Python with Black and Ruff for readable GitHub diffs.
- Added fixture-backed pytest coverage for the CLI entrypoint, schema validation,
  factor-source modes, academic-factor wiring, brief brevity, data-quality gates,
  risk-method separation, and dashboard loading.
- Added a compact fixture dataset under `tests/fixtures`.
- Added GitHub Actions CI for Python 3.11, Ruff, Black, and pytest.
- Updated documentation to mark loan proxies as experimental and regime-break
  monitoring as a hybrid transition/stress framework.

## 0.6.0 - Historical Validity And Robustness

- Added Kenneth French factor ingestion to extend academic factor history before
  ETF proxy inception.
- Preserved tradeable ETF proxies separately from academic factor portfolios and
  exposed factor provenance in outputs.
- Added regime-break risk monitoring based on historical transitions, severity
  percentiles, distance-to-risk, and confidence.
- Added simple validation baselines: previous winner, factor momentum, logistic
  regression, interpretable decision tree, equal-weight factors, and SPY.
- Added a canonical `RunResult` schema contract and lightweight schema validation test.
- Added data-quality gates that reduce confidence and mark runs as data impaired when critical series fail or go stale.
- Kept validation, diagnostics, and data-health detail in the dashboard while
  preserving the daily brief's concise format.
- Stabilized v0.6 contracts by wiring `RunResult` construction into the agent,
  adding explicit factor-source modes, and labeling stress-percentile versus
  regime-break risk outputs.
- Hardened v0.6 with Ruff/Black formatting, transition-window risk-ranking tests,
  academic-mode validation tests, and baseline value-add contract checks.

## 0.5.0 - Validation Framework

- Added multi-horizon walk-forward validation for 1M, 3M, and 6M factor leadership.
- Compared model-selected factor exposure against equal-weight factors and SPY.
- Added hit rate, average forward excess return, information ratio, max drawdown,
  turnover proxy, and confusion matrices.
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
