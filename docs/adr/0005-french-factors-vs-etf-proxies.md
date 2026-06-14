# ADR 0005: Kenneth French Academic Factors And ETF Tradeable Proxies

## Status

Accepted

## Context

ETF factor proxies are useful for tradeable implementation, but many factor ETFs
have limited histories. Short histories weaken regime validation, historical
analogs, and factor-leadership research.

The Kenneth French Data Library provides longer academic factor histories for
common equity factors. These series are research portfolios, not directly
tradeable instruments.

## Decision

Use Kenneth French academic factor series to extend historical research coverage
before ETF proxy inception. Preserve ETF proxy returns as separate tradeable
series and let ETF data override academic data when both are available.

Label factor provenance in `run_result.json` so consumers can distinguish
academic history from tradeable proxy observations.

## Consequences

- Historical validation and analogs gain more observations.
- Current exposure and future portfolio workflows remain anchored to tradeable proxies.
- Low volatility remains ETF-only until a well-documented academic proxy is added.
- Reported research performance must not be described as directly investable unless it is based on tradeable proxy data.
