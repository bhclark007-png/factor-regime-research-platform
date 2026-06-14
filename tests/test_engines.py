from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import dashboard
from factor_agent.agent import _select_factor_returns, run
from factor_agent.brief import make_daily_brief
from factor_agent.features import build_factor_excess_returns
from factor_agent.quality import evaluate_data_quality
from factor_agent.risk import dynamic_regime_risks, regime_break_risk_monitor
from factor_agent.schema import validate_run_result


def test_run_agent_smoke_uses_fixture_data(
    tmp_path: Path,
    sample_macro: pd.DataFrame,
    sample_prices: pd.DataFrame,
    sample_french_history,
    source_statuses,
    fake_train_result: dict,
) -> None:
    with (
        patch(
            "factor_agent.agent.get_fred", return_value=(sample_macro, source_statuses)
        ),
        patch(
            "factor_agent.agent.get_etf_prices",
            return_value=(sample_prices, source_statuses),
        ),
        patch(
            "factor_agent.agent.get_kenneth_french_factors",
            return_value=sample_french_history,
        ),
        patch("factor_agent.agent.train_factor_model", return_value=fake_train_result),
        patch(
            "factor_agent.agent.validate_factor_model",
            return_value={"version": "0.6.1", "by_horizon": {}, "summary": []},
        ),
    ):
        payload = run(
            start="2020-01-01",
            end=None,
            horizon=3,
            output_dir=str(tmp_path),
            run_id="engine_fixture",
            factor_source="tradeable",
        )

    validate_run_result(payload)
    validate_run_result(
        json.loads(
            (tmp_path / "latest" / "run_result.json").read_text(encoding="utf-8")
        )
    )


def test_run_result_schema_fixture_validates(fixture_run_result_path: Path) -> None:
    payload = json.loads(fixture_run_result_path.read_text(encoding="utf-8"))

    validate_run_result(payload)


def test_factor_source_modes_are_distinct(
    sample_prices: pd.DataFrame,
    sample_french_returns: pd.DataFrame,
) -> None:
    tradeable_returns = build_factor_excess_returns(sample_prices)

    academic, academic_meta = _select_factor_returns(
        "academic", sample_french_returns, tradeable_returns
    )
    tradeable, tradeable_meta = _select_factor_returns(
        "tradeable", sample_french_returns, tradeable_returns
    )
    combined, combined_meta = _select_factor_returns(
        "combined", sample_french_returns, tradeable_returns
    )

    assert academic_meta["selected_mode"] == "academic"
    assert tradeable_meta["selected_mode"] == "tradeable"
    assert combined_meta["selected_mode"] == "combined"
    assert academic["value"].iloc[0] == sample_french_returns["value"].iloc[0]
    assert "low_vol" in tradeable.columns
    assert "low_vol" in combined.columns


def test_brief_generation_stays_concise() -> None:
    brief = make_daily_brief(
        probabilities=pd.Series({"value": 0.45, "quality": 0.35, "momentum": 0.20}),
        cv_accuracy=0.4,
        credit_score=62,
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
    assert len(brief.splitlines()) < 90


def test_data_quality_gates_lower_confidence() -> None:
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


def test_risk_engine_reports_honest_method_fields() -> None:
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
    regime_break = regime_break_risk_monitor(features, winners, "Risk-On Expansion")

    assert stress["method"] == "stress_percentile_monitoring"
    assert regime_break["method"] == "historical_regime_break_monitoring"
    assert regime_break["defined_break_count"] > 0
    assert regime_break["top_regime_change_risks"]


def test_dashboard_loads_latest_run_result(
    monkeypatch,
    fixture_run_result_path: Path,
) -> None:
    monkeypatch.setattr(dashboard, "LATEST_JSON", fixture_run_result_path)

    result = dashboard.load_result()

    assert result["schema_version"] == "0.6.1"
    assert result["run_id"] == "fixture"
