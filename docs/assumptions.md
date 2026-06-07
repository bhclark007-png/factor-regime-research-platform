# Model Assumptions

## General

- The system is decision support, not an autonomous trading engine.
- Public data is acceptable for research, but source gaps and revisions must be surfaced.
- Interpretability is preferred over marginal gains from opaque complexity.
- ETF proxies are imperfect but useful for early factor-regime research.

## Data

- Monthly features are built from the last available observation in each month.
- Missing macro data is forward/back filled for model training only after source status is recorded.
- Cached data may be used when live feeds fail, and this is recorded in `run_result.json`.
- `ccc_oas` can use a local seed file plus latest FRED updates.
- `ism_mfg` can use a manufacturing-growth proxy when official PMI history is unavailable.

## Scores

- Credit Leadership Score is bounded from 0 to 100.
- Regime Stability Score is bounded from 0 to 100.
- Existing static score thresholds are provisional and must be reviewed as validation improves.
- Dynamic risk monitoring ranks indicators by historical percentile stress rather than fixed thresholds where possible.

## Recalibration

- Daily: update data and regenerate outputs.
- Weekly: refresh analog analysis.
- Monthly: review risk-factor behavior.
- Quarterly: review model coefficients/features.
- Annually: review methodology and assumptions.
