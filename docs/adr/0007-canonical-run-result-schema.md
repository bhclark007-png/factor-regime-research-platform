# ADR 0007: Canonical RunResult Schema

## Status

Accepted

## Context

The backend, dashboard, daily brief writer, and future portfolio module all need the same understanding of a completed run. Without a canonical contract, changes to `run_result.json` can silently break downstream consumers.

## Decision

Define a typed `RunResult` schema module and validate the emitted `run_result.json` payload before writing final artifacts.

The schema includes regime output, factor probabilities, credit drivers, risks, validation, factor-history provenance, data-quality status, source health, and artifact paths.

## Consequences

- Dashboard and brief consumers can depend on one output contract.
- Future portfolio modules can read the same run artifact without duplicating assumptions.
- Schema validation remains lightweight for now and should become stricter as external consumers are added.
- Major contract changes require changelog updates and, when architectural, a new ADR.
