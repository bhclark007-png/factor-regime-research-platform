from __future__ import annotations

import pandas as pd

from factor_agent.backtest import (
    factor_backtest_metrics,
    validate_credit_leadership,
    validate_factor_model,
)


def test_backtest_metrics_contract() -> None:
    index = pd.date_range("2023-01-31", periods=6, freq="ME")
    forward = pd.DataFrame({"value": [0.1, -0.1, 0.2, 0.0, 0.1, -0.05]}, index=index)
    winner = pd.Series(["value"] * 6, index=index, name="winner")

    metrics = factor_backtest_metrics(forward, winner)

    assert metrics["observations"] == 6
    assert metrics["factor_metrics"][0]["factor"] == "value"


def test_validation_contract() -> None:
    index = pd.date_range("2018-01-31", periods=84, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas": [300 + (i % 12) for i in range(84)],
            "vix": [15 + (i % 9) for i in range(84)],
            "cpi_3m_ann": [2 + (i % 4) * 0.2 for i in range(84)],
            "ism_mfg": [48 + (i % 8) for i in range(84)],
        },
        index=index,
    )
    factor_excess = pd.DataFrame(
        {
            "value": [0.01 if i % 3 == 0 else -0.002 for i in range(84)],
            "quality": [0.008 if i % 3 == 1 else 0.001 for i in range(84)],
            "momentum": [0.009 if i % 3 == 2 else -0.001 for i in range(84)],
        },
        index=index,
    )

    validation = validate_factor_model(
        features, factor_excess, horizons=(1, 3), min_train_months=36
    )
    compared = {
        row["baseline"] for row in validation["by_horizon"]["1"]["model_value_add"]
    }

    assert validation["version"] == "0.6.1"
    assert "validation_window" in validation
    assert "1" in validation["by_horizon"]
    assert "confusion_matrix" in validation["by_horizon"]["1"]
    assert "baselines" in validation["by_horizon"]["1"]
    assert "model_value_add" in validation["by_horizon"]["1"]
    assert "previous_winner" in compared
    assert "equal_weight_factors" in compared
    assert "spy" in compared
    assert validation["model"] == "random_forest"
    assert "credit_leadership_validation" in validation


def test_credit_leadership_validation_ranks_credit_variables() -> None:
    index = pd.date_range("2017-01-31", periods=72, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas": [300 if i % 2 == 0 else 500 for i in range(72)],
            "ig_oas": [100 if i % 2 == 0 else 150 for i in range(72)],
            "ccc_oas": [800 if i % 2 == 0 else 1100 for i in range(72)],
            "hy_oas_3m_chg": [-20 if i % 2 == 0 else 60 for i in range(72)],
            "vix": [15 if i % 2 == 0 else 28 for i in range(72)],
        },
        index=index,
    )
    winner = pd.Series(
        ["value" if i % 2 == 0 else "quality" for i in range(72)],
        index=index,
        name="winner",
    )

    validation = validate_credit_leadership(features, winner)

    assert validation["method"] == "mean_spread_and_one_vs_rest_correlation"
    assert validation["top_variables"]
    assert validation["top_variables"][0]["explanatory_value"] > 0
