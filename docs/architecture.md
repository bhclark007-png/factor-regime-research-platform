# Architecture

## Data Flow

1. Data layer downloads or loads macro, credit, volatility, and ETF series.
2. Feature layer converts raw daily/monthly series into monthly regime features.
3. Factor layer selects tradeable ETF factors, Kenneth French academic factors,
   or combined academic-then-tradeable history.
4. Model layer trains a time-series-aware classifier for likely factor leadership.
5. Scoring engines calculate credit leadership, stability, dynamic risks, analogs, and backtest metrics.
6. Brief generator writes a concise Markdown brief.
7. Artifact writer emits JSON, CSV, run history, and latest pointers.
8. Dashboard reads `output/latest/run_result.json`.

## Regime Engine

The current regime engine combines model-implied top factor, credit leadership, and stability into interpretable labels:

- Risk-On Expansion
- Reflation / Cyclical
- Defensive / Slowdown
- Transition / Risk-Off Watch
- Mixed / Data-Dependent

Future work should replace label heuristics with a validated probabilistic
regime classifier while preserving interpretability.

## Factor Engine

The factor engine scores:

- Momentum
- Quality
- Value
- Low Volatility
- Small Cap

Current implementation supports three factor-source modes:

- `tradeable`: ETF proxies for current implementation.
- `academic`: Kenneth French academic factors for longer historical validation.
- `combined`: Kenneth French history before ETF availability, then ETF proxies.

Low-volatility remains tradeable-only until a defensible academic proxy is added.

## Credit Leadership Engine

Credit leadership uses HY OAS, IG OAS, CCC spreads, and relative lower-quality
credit behavior. It returns a 0-100 score and drivers. Loan proxies are
experimental and are not yet core model inputs.

## Analog Engine

The analog engine standardizes macro/credit features and finds historical
environments closest to the latest observation. It reports subsequent factor
performance for those periods.

## Brief Generator

The brief is designed to be read in under two minutes:

1. BLUF
2. Why
3. Regime risks
4. Stability score
5. Historical analog summary

## Future Portfolio Module

The future portfolio module should consume factor recommendations and compare
them to user holdings. It should not be coupled to data ingestion or model
training.
