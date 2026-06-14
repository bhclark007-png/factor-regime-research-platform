from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from factor_agent.agent import run
from factor_agent.schema import validate_run_result


def test_run_agent_completes_with_fixture_data(
    tmp_path: Path,
    sample_macro,
    sample_prices,
    sample_french_history,
    source_statuses,
    fake_train_result,
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
            run_id="fixture_run",
            factor_source="tradeable",
        )

    validate_run_result(payload)
    latest_json = tmp_path / "latest" / "run_result.json"
    assert latest_json.exists()
    validate_run_result(json.loads(latest_json.read_text(encoding="utf-8")))


def test_fixture_run_result_validates_against_schema(
    fixture_run_result_path: Path,
) -> None:
    payload = json.loads(fixture_run_result_path.read_text(encoding="utf-8"))
    validate_run_result(payload)


def test_run_agent_py_entrypoint_completes_with_fixture_data(
    tmp_path: Path,
    sample_macro,
    sample_prices,
    sample_french_history,
    source_statuses,
    fake_train_result,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent.py",
            "--output",
            str(tmp_path),
            "--run-id",
            "entrypoint_fixture",
            "--factor-source",
            "tradeable",
        ],
    )
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
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "run_agent.py"),
            run_name="__main__",
        )

    validate_run_result(
        json.loads(
            (tmp_path / "latest" / "run_result.json").read_text(encoding="utf-8")
        )
    )
