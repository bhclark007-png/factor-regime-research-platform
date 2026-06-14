from __future__ import annotations

from unittest.mock import patch

from factor_agent.agent import _select_factor_returns, run
from factor_agent.features import build_factor_excess_returns


def test_factor_source_modes_select_expected_series(
    sample_prices,
    sample_french_returns,
) -> None:
    tradeable = build_factor_excess_returns(sample_prices)

    academic, academic_meta = _select_factor_returns(
        "academic", sample_french_returns, tradeable
    )
    tradeable_selected, tradeable_meta = _select_factor_returns(
        "tradeable", sample_french_returns, tradeable
    )
    combined, combined_meta = _select_factor_returns(
        "combined", sample_french_returns, tradeable
    )

    assert academic_meta["selected_mode"] == "academic"
    assert tradeable_meta["selected_mode"] == "tradeable"
    assert combined_meta["selected_mode"] == "combined"
    assert set(academic.columns) == set(sample_french_returns.columns)
    assert "low_vol" in tradeable_selected.columns
    assert "low_vol" in combined.columns


def test_kenneth_french_used_in_academic_and_combined_modes(
    tmp_path,
    sample_macro,
    sample_prices,
    sample_french_history,
    source_statuses,
    fake_train_result,
) -> None:
    captured = {}

    def capture_validation(_, factor_excess):
        captured["columns"] = list(factor_excess.columns)
        captured["first_value"] = float(factor_excess["value"].iloc[0])
        return {"version": "0.6.1", "by_horizon": {}, "summary": []}

    for mode in ["academic", "combined"]:
        captured.clear()
        with (
            patch(
                "factor_agent.agent.get_fred",
                return_value=(sample_macro, source_statuses),
            ),
            patch(
                "factor_agent.agent.get_etf_prices",
                return_value=(sample_prices, source_statuses),
            ),
            patch(
                "factor_agent.agent.get_kenneth_french_factors",
                return_value=sample_french_history,
            ),
            patch(
                "factor_agent.agent.train_factor_model", return_value=fake_train_result
            ),
            patch(
                "factor_agent.agent.validate_factor_model",
                side_effect=capture_validation,
            ),
        ):
            payload = run(
                start="2020-01-01",
                end=None,
                horizon=3,
                output_dir=str(tmp_path / mode),
                run_id=f"{mode}_run",
                factor_source=mode,
            )

        assert payload["factor_history"]["selected_mode"] == mode
        assert captured["first_value"] == sample_french_history.returns["value"].iloc[0]
        assert "value" in captured["columns"]
