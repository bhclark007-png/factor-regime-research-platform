from __future__ import annotations

import pandas as pd

from factor_agent.scores import (
    confidence_breakdown,
    credit_leadership_score,
    regime_label,
    regime_stability_score,
)


def test_scores_are_bounded() -> None:
    latest = pd.Series(
        {
            "hy_oas": 300,
            "hy_oas_1m_chg": -20,
            "hy_oas_3m_chg": -40,
            "ccc_oas_3m_chg": -80,
            "ccc_minus_hy_3m_chg": -60,
            "vix": 18,
            "ism_mfg": 52,
            "cpi_3m_ann": 2.5,
        }
    )

    credit, drivers = credit_leadership_score(latest)
    stability, risks = regime_stability_score(latest)

    assert 0 <= credit <= 100
    assert 0 <= stability <= 100
    assert drivers
    assert isinstance(risks, list)


def test_regime_label_is_stable() -> None:
    probabilities = pd.Series({"value": 0.55, "quality": 0.25, "momentum": 0.20})

    assert (
        regime_label(probabilities, credit_score=80, stability=80)
        == "Risk-On Expansion"
    )
    assert (
        regime_label(probabilities, credit_score=50, stability=30)
        == "Transition / Risk-Off Watch"
    )


def test_confidence_breakdown_uses_interpretable_components() -> None:
    probabilities = pd.Series({"value": 0.55, "quality": 0.25, "momentum": 0.20})
    analogs = {
        "analogs": [
            {"best_factor": "value"},
            {"best_factor": "quality"},
            {"best_factor": "value"},
        ]
    }

    confidence = confidence_breakdown(
        probabilities,
        analogs,
        credit_score=70,
        stability_score=80,
        data_quality={"confidence_multiplier": 0.8},
    )

    assert confidence["method"] == "weighted_interpretable_components"
    assert 0 <= confidence["score"] <= 1
    assert confidence["components"]["analog_agreement"] == 2 / 3
