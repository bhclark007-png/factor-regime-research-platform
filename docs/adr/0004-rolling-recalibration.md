# ADR 0004: Use Scheduled Recalibration

## Status

Accepted

## Context

Daily model recalibration can create noise, instability, and false precision.

## Decision

Use daily data updates, weekly analog updates, monthly risk-factor updates, quarterly model reviews, and annual methodology reviews.

## Consequences

Outputs remain current without overfitting daily noise. The model lifecycle is explicit and easier to audit.
