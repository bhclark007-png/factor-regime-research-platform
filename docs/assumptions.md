# Model Assumptions

## General

- The system is decision support, not an autonomous trading engine.
- Public data is acceptable for research, but source gaps and revisions must be
  surfaced.
- Interpretability is preferred over marginal gains from opaque complexity.
- ETF proxies are imperfect but useful for early factor-regime research.

## Data

- Monthly features are built from the last available observation in each month.
- Missing macro data is forward/back filled for model training only after source status is recorded.
- Cached data may be used when live feeds fail, and this is recorded in `run_result.json`.
- `ccc_oas` can use a local seed file plus latest FRED updates.
- `ism_mfg` can use a manufacturing-growth proxy when official PMI history is unavailable.
- Kenneth French factors are academic long-short or portfolio factor series.
  They improve historical research coverage but are not treated as directly
  tradeable products.
- ETF factor proxies are the preferred tradeable series for current exposure and future portfolio modules.
- When both academic and ETF factor data exist for the same month, ETF proxy
  returns take precedence in the combined factor history.
- Daily runs default to `tradeable` factor-source mode unless
  `--factor-source academic` or `--factor-source combined` is explicitly selected.
- Loan proxies are experimental and are not treated as reliable core inputs yet.
- Critical source failures or stale critical observations reduce model
  confidence. If any critical source fails or too many critical series are stale,
  the run is marked data impaired.

## Scores

- Credit Leadership Score is bounded from 0 to 100.
- Regime Stability Score is bounded from 0 to 100.
- Existing static score thresholds are provisional and must be reviewed as validation improves.
- Dynamic risk monitoring ranks indicators by historical regime-break behavior
  and severity percentile rather than fixed thresholds where possible.
- Regime-break risk logic is hybrid. It uses realized factor-leadership changes
  filtered by credit or volatility stress, but it is not yet a fully validated
  transition-label model.
- Validation baselines are intentionally simple; the Random Forest model should
  be judged against them before adding complexity.

## Recalibration

- Daily: update data and regenerate outputs.
- Weekly: refresh analog analysis.
- Monthly: review risk-factor behavior.
- Quarterly: review model coefficients/features.
- Annually: review methodology and assumptions.
