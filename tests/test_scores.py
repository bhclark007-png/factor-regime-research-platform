from __future__ import annotations

import pandas as pd

from factor_agent.scores import (
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
