from __future__ import annotations

import json
from pathlib import Path

from factor_agent.schema import validate_run_result


def test_minimal_run_result_schema_validation() -> None:
    payload = {
        "schema_version": "0.6.1",
        "run_id": "test",
        "generated_at": "2026-06-13T12:00:00",
        "parameters": {},
        "regime": {
            "label": "Mixed",
            "top_factor": "value",
            "top_factor_probability": 0.4,
            "adjusted_confidence": 0.3,
        },
        "factor_probabilities": [],
        "data_status": {},
        "data_quality": {"data_impaired": False},
        "factor_history": {"selected_mode": "tradeable"},
        "validation": {"version": "0.6.1"},
        "dynamic_risks": {"method": "stress_percentile_monitoring"},
        "regime_break_risks": {"method": "historical_regime_break_monitoring"},
        "artifacts": {},
    }

    validate_run_result(payload)


def test_fixture_run_result_validates_against_schema(
    fixture_run_result_path: Path,
) -> None:
    payload = json.loads(fixture_run_result_path.read_text(encoding="utf-8"))
    validate_run_result(payload)
