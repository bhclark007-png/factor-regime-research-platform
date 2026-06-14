from __future__ import annotations

import pandas as pd

from factor_agent.analog import find_historical_analogs
from factor_agent.risk import (
    dynamic_regime_risks,
    identify_regime_breaks,
    regime_break_risk_monitor,
)


def test_analog_and_dynamic_risk_contracts() -> None:
    index = pd.date_range("2020-01-31", periods=50, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas_3m_chg": range(50),
            "vix_1m_chg": [x % 10 for x in range(50)],
            "ism_3m_chg": [5 - (x % 8) for x in range(50)],
            "cpi_3m_ann": [2 + (x % 5) * 0.1 for x in range(50)],
        },
        index=index,
    )
    forward = pd.DataFrame(
        {
            "value": [0.01] * 50,
            "quality": [0.005] * 50,
            "momentum": [-0.002] * 50,
        },
        index=index,
    )
    break_winner = pd.Series(
        (["value", "value", "quality", "quality"] * 13)[:50], index=index
    )

    analogs = find_historical_analogs(features, forward, n=3)
    risks = dynamic_regime_risks(features)
    break_risks = regime_break_risk_monitor(features, break_winner, "test")

    assert "analogs" in analogs
    assert "risks" in risks
    assert "top_regime_change_risks" in break_risks


def test_risk_outputs_are_self_describing() -> None:
    index = pd.date_range("2015-01-31", periods=96, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas_3m_chg": [i % 20 for i in range(96)],
            "vix_1m_chg": [i % 15 for i in range(96)],
            "ccc_minus_hy_3m_chg": [i % 10 for i in range(96)],
            "ism_3m_chg": [10 - (i % 12) for i in range(96)],
        },
        index=index,
    )
    winner = pd.Series((["value"] * 6 + ["quality"] * 6) * 8, index=index)

    stress = dynamic_regime_risks(features)
    breaks = regime_break_risk_monitor(features, winner, "test")

    assert stress["method"] == "stress_percentile_monitoring"
    assert breaks["method"] == "historical_regime_break_monitoring"
    assert "defined_break_count" in breaks
    for risk in breaks["top_regime_change_risks"]:
        assert "historical_frequency_before_transitions" in risk
        assert "severity_percentile" in risk


def test_regime_break_ranking_uses_transition_windows() -> None:
    index = pd.date_range("2010-01-31", periods=120, freq="ME")
    winners = pd.Series(
        (["value"] * 12 + ["quality"] * 12) * 5,
        index=index,
        name="winner",
    )
    hy_stress = pd.Series(0.0, index=index)
    vix_stress = pd.Series(0.0, index=index)

    for block_start in range(0, len(index), 12):
        block = index[block_start : block_start + 12]
        if len(block) < 12:
            continue
        hy_stress.loc[block[9:12]] = 100.0
        if block_start > 0:
            hy_stress.loc[block[0]] = 100.0
        vix_stress.loc[block[2:6]] = 100.0

    features = pd.DataFrame(
        {
            "hy_oas_3m_chg": hy_stress,
            "vix_1m_chg": vix_stress,
        },
        index=index,
    )

    breaks = identify_regime_breaks(winners, features)
    risks = regime_break_risk_monitor(features, winners, "test", top_n=2)
    top_risk = risks["top_regime_change_risks"][0]
    second_risk = risks["top_regime_change_risks"][1]

    assert len(breaks) > 0
    assert top_risk["indicator"] == "hy_oas_3m_chg"
    assert (
        top_risk["historical_frequency_before_transitions"]
        > second_risk["historical_frequency_before_transitions"]
    )


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
