# ADR 0006: Regime-Break-Based Risk Monitoring

## Status

Accepted

## Context

Percentile-only risk flags are easy to interpret, but they can overstate routine extremes and miss the indicators that historically mattered before a regime transition.

The platform needs regime-risk logic that is still interpretable but more historically grounded.

## Decision

Supplement percentile stress monitoring with regime-break risk monitoring. Define historical breaks using changes in realized factor leadership, credit stress, volatility, and model regime labels where available.

For the active regime, rank indicators by how frequently they appeared before historical breaks and report historical frequency, current distance-to-risk, severity percentile, and confidence.

## Consequences

- Risk warnings become more tied to observed historical transition behavior.
- The dashboard can show diagnostics without expanding the daily brief.
- Break definitions remain research assumptions and should be reviewed as more validation history is added.
- Percentile severity remains useful context, but it is no longer the only risk-ranking mechanism.
