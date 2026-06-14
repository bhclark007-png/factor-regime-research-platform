from __future__ import annotations

import pandas as pd

from factor_agent.brief import make_daily_brief
from factor_agent.quality import evaluate_data_quality
from factor_agent.risk import (
    dynamic_regime_risks,
    identify_regime_breaks,
    regime_break_risk_monitor,
)


def test_daily_brief_remains_concise_without_validation_tables() -> None:
    brief = make_daily_brief(
        probabilities=pd.Series({"value": 0.5, "quality": 0.3, "momentum": 0.2}),
        cv_accuracy=0.4,
        credit_score=60,
        credit_drivers=["HY spreads are contained."],
        stability_score=70,
        risks=[],
        dynamic_risks={"transition_probability": None, "risks": []},
        regime="Risk-On Expansion",
        analogs={"analogs": []},
        data_quality={"data_impaired": False},
        feature_importances=pd.Series({"hy_oas": 0.2}),
    )

    assert "Validation" not in brief
    assert "confusion" not in brief.lower()
    assert "model_value_add" not in brief
    assert len(brief.splitlines()) < 90


def test_data_quality_gates_lower_confidence_for_failed_or_stale_critical_data() -> (
    None
):
    quality = evaluate_data_quality(
        [
            {
                "name": "hy_oas",
                "ticker": "BAMLH0A0HYM2",
                "status": "failed",
                "latest_observation": None,
            },
            {
                "name": "SPY",
                "ticker": "SPY",
                "status": "cache",
                "latest_observation": "2026-01-01",
            },
        ],
        as_of="2026-06-13",
    )

    assert quality["data_impaired"] is True
    assert quality["confidence_multiplier"] < 1.0
    assert quality["issues"]


def test_risk_engine_distinguishes_stress_percentile_and_regime_break_methods() -> None:
    index = pd.date_range("2010-01-31", periods=120, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas_3m_chg": [0.0] * 120,
            "vix_1m_chg": [0.0] * 120,
        },
        index=index,
    )
    winners = pd.Series((["value"] * 12 + ["quality"] * 12) * 5, index=index)
    for block_start in range(12, len(index), 12):
        features.loc[index[block_start], "hy_oas_3m_chg"] = 100.0
        features.loc[index[block_start - 2 : block_start], "hy_oas_3m_chg"] = 100.0

    stress = dynamic_regime_risks(features)
    breaks = identify_regime_breaks(winners, features)
    regime_break = regime_break_risk_monitor(features, winners, "test")

    assert stress["method"] == "stress_percentile_monitoring"
    assert regime_break["method"] == "historical_regime_break_monitoring"
    assert len(breaks) > 0
    assert regime_break["defined_break_count"] == len(breaks)
    assert regime_break["top_regime_change_risks"][0]["break_observations"] > 0
