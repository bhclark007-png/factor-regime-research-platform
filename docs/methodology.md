# Methodology

## Data Transformations

- Raw series are resampled monthly using month-end last observation.
- Credit spreads are transformed into levels and 1/3/6-month changes.
- Yield-curve slope is calculated as 10Y minus 2Y.
- Inflation momentum is calculated using annualized 3- and 6-month CPI changes.
- Factor ETF returns are monthly returns minus SPY monthly returns.

## Backtesting

Primary forecast horizons:

- 1 month
- 3 months
- 6 months

Current prototype uses a configurable horizon and writes summary metrics. Future work should store separate walk-forward prediction histories for each horizon.

Tracked metrics:

- Hit rate
- Precision / win rate
- Recall, once explicit class prediction histories exist
- Forward excess returns
- Max drawdown

## Validation Framework

- Use time-series splits only.
- Avoid random train/test splits.
- Report source coverage and missing-feed status with every run.
- Compare model results against simple baselines.
- Track whether improvements persist out of sample and after transaction-cost assumptions.

## Recalibration Process

- Daily data updates do not imply daily model recalibration.
- Weekly analog updates can refresh similarity analysis.
- Monthly risk-factor updates can refresh transition-monitoring indicators.
- Quarterly model reviews can adjust features or coefficients.
- Annual methodology review should revisit all assumptions, thresholds, and data vendors.
