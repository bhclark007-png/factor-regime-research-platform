# Methodology

## Data Transformations

- Raw series are resampled monthly using month-end last observation.
- Credit spreads are transformed into levels and 1/3/6-month changes.
- Yield-curve slope is calculated as 10Y minus 2Y.
- Inflation momentum is calculated using annualized 3- and 6-month CPI changes.
- Factor ETF returns are monthly returns minus SPY monthly returns.
- Kenneth French academic factors are loaded as monthly decimal returns and mapped to the platform factor names where a close research analogue exists: SMB to small cap, HML to value, momentum to momentum, and an RMW/CMA quality composite to quality.
- Academic factors extend research history before ETF proxy inception. ETF proxy returns override academic series once tradeable proxy data is available. Low-volatility history remains ETF-only because the Kenneth French library does not provide a direct low-volatility analogue in the current implementation.

## Backtesting

Primary forecast horizons:

- 1 month
- 3 months
- 6 months

Current prototype uses a configurable horizon and writes summary metrics. Future work should store separate walk-forward prediction histories for each horizon.

The validation module runs expanding-window model validation across 1M, 3M, and 6M horizons. Each validation observation trains only on prior months, selects the highest-probability factor, and compares that selected factor's forward excess return with:

- equal-weight factor exposure
- SPY baseline, represented as zero factor excess return
- naive previous-winner factor selection
- six-month factor momentum
- multinomial logistic regression
- shallow interpretable decision tree
- the realized best factor for confusion-matrix analysis

Dashboard validation defaults to the most recent 120 monthly observations and samples validation points every three months to keep daily runs practical. Longer research-window validation can be run as a separate research job when model changes warrant it.

Tracked metrics:

- Hit rate
- Average forward excess return
- Information ratio
- Forward excess returns
- Max drawdown
- Turnover proxy
- Confusion matrix

## Validation Framework

- Use time-series splits only.
- Avoid random train/test splits.
- Report source coverage and missing-feed status with every run.
- Compare model results against simple baselines.
- Track whether improvements persist out of sample and after transaction-cost assumptions.
- Report whether the current Random Forest model adds value over simple baselines before treating model-selected leadership as useful.

## Regime-Break Risk Framework

- Historical regime breaks are defined using changes in realized factor leadership, stress indicators, and model regime labels where available.
- For each active regime, monitored indicators are ranked by how often they appeared before historical breaks.
- Each risk reports historical frequency before transitions, current distance-to-risk, severity percentile, and confidence.
- Percentile-only monitoring is retained as context, but regime-break frequency is the preferred interpretability layer.

## Recalibration Process

- Daily data updates do not imply daily model recalibration.
- Weekly analog updates can refresh similarity analysis.
- Monthly risk-factor updates can refresh transition-monitoring indicators.
- Quarterly model reviews can adjust features or coefficients.
- Annual methodology review should revisit all assumptions, thresholds, and data vendors.
