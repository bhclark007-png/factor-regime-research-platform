from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RegimeResult:
    label: str
    top_factor: str
    top_factor_probability: float
    adjusted_confidence: float
    credit_leadership_score: int
    regime_stability_score: int
    cv_accuracy: float


@dataclass
class RunResult:
    run_id: str
    generated_at: str
    parameters: dict[str, Any]
    regime: RegimeResult
    factor_probabilities: list[dict[str, Any]]
    credit_drivers: list[str]
    risks: list[dict[str, Any]]
    dynamic_risks: dict[str, Any]
    regime_break_risks: dict[str, Any]
    historical_analogs: dict[str, Any]
    feature_importances: list[dict[str, Any]]
    backtest_summary: list[dict[str, Any]]
    backtest_metrics: dict[str, Any]
    validation: dict[str, Any]
    factor_history: dict[str, Any]
    data_quality: dict[str, Any]
    data_status: dict[str, Any]
    artifacts: dict[str, Any]
    schema_version: str = "0.6.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_run_result(payload: dict[str, Any]) -> None:
    """Lightweight schema validation for run_result.json."""
    required = {
        "schema_version",
        "run_id",
        "generated_at",
        "parameters",
        "regime",
        "factor_probabilities",
        "data_status",
        "data_quality",
        "factor_history",
        "validation",
        "artifacts",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"RunResult missing required fields: {missing}")

    regime_required = {
        "label",
        "top_factor",
        "top_factor_probability",
        "adjusted_confidence",
    }
    regime_missing = sorted(regime_required - set(payload["regime"]))
    if regime_missing:
        raise ValueError(f"RunResult.regime missing required fields: {regime_missing}")

    if not isinstance(payload["factor_probabilities"], list):
        raise ValueError("RunResult.factor_probabilities must be a list")
    if not isinstance(payload["data_quality"].get("data_impaired"), bool):
        raise ValueError("RunResult.data_quality.data_impaired must be a bool")
    if payload["schema_version"] != "0.6.1":
        raise ValueError("RunResult.schema_version must be 0.6.1")
    if payload.get("validation", {}).get("version") != "0.6.1":
        raise ValueError("RunResult.validation.version must be 0.6.1")

    factor_history = payload["factor_history"]
    if factor_history.get("selected_mode") not in {"academic", "tradeable", "combined"}:
        raise ValueError(
            "RunResult.factor_history.selected_mode must be academic, tradeable, or combined"
        )

    dynamic_risks = payload.get("dynamic_risks", {})
    if dynamic_risks and dynamic_risks.get("method") != "stress_percentile_monitoring":
        raise ValueError(
            "RunResult.dynamic_risks.method must describe stress-percentile monitoring"
        )

    break_risks = payload.get("regime_break_risks", {})
    if (
        break_risks
        and break_risks.get("method") != "historical_regime_break_monitoring"
    ):
        raise ValueError(
            "RunResult.regime_break_risks.method must describe historical regime-break monitoring"
        )
