from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from factor_agent.signal_validation import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    calculate_forward_excess_returns,
    calculate_hit_rate,
    calculate_information_coefficient,
    classify_signal_strength,
    validate_all_signals,
    validate_single_signal,
)


def test_information_coefficient_handles_rank_relationship() -> None:
    feature = pd.Series([1, 2, 3, 4, 5])
    target = pd.Series([2, 4, 6, 8, 10])

    assert calculate_information_coefficient(feature, target) == pytest.approx(1.0)


def test_information_coefficient_handles_nan_and_constant_inputs() -> None:
    feature = pd.Series([1, 1, 1, np.nan, 1])
    target = pd.Series([1, 2, 3, 4, 5])

    assert math.isnan(calculate_information_coefficient(feature, target))


def test_hit_rate_directional_logic() -> None:
    feature = pd.Series([1, 2, 3, 4, 5])
    target = pd.Series([-0.03, -0.01, 0.0, 0.02, 0.04])

    assert calculate_hit_rate(feature, target) == 1.0


def test_forward_return_alignment_avoids_lookahead() -> None:
    index = pd.date_range("2024-01-31", periods=5, freq="ME")
    returns = pd.DataFrame({"momentum": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=index)

    forward = calculate_forward_excess_returns(returns, horizons=(1, 3))

    assert forward[1].loc[index[0], "momentum"] == returns.loc[index[1], "momentum"]
    assert forward[3].loc[index[0], "momentum"] == returns.iloc[1:4][
        "momentum"
    ].sum()


def test_validate_single_signal_returns_required_metrics() -> None:
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    feature = pd.Series(range(36), index=index, name="hy_oas_3m_chg")
    returns = pd.DataFrame(
        {
            "momentum": [0.01 if i > 18 else -0.01 for i in range(36)],
            "value": [-0.005 if i > 18 else 0.005 for i in range(36)],
        },
        index=index,
    )
    forward = calculate_forward_excess_returns(returns, horizons=(1,))

    summary, detail = validate_single_signal("hy_oas_3m_chg", feature, forward)

    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(DETAIL_COLUMNS).issubset(detail.columns)
    assert not summary.empty
    assert not detail.empty


def test_validate_all_signals_returns_non_empty_summary_and_detail() -> None:
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    features = pd.DataFrame(
        {
            "hy_oas_3m_chg": range(36),
            "vix_1m_chg": [i % 5 for i in range(36)],
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "momentum": [0.01 if i % 2 == 0 else -0.002 for i in range(36)],
            "quality": [-0.002 if i % 2 == 0 else 0.01 for i in range(36)],
        },
        index=index,
    )
    forward = calculate_forward_excess_returns(returns, horizons=(1, 3))

    summary, detail = validate_all_signals(features, forward, horizons=(1, 3))

    assert not summary.empty
    assert not detail.empty
    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(DETAIL_COLUMNS).issubset(detail.columns)
    assert set(summary["horizon"].unique()) == {1, 3}


def test_classify_signal_strength_returns_expected_labels() -> None:
    keep = {
        "sample_count": 60,
        "information_coefficient": 0.15,
        "hit_rate": 0.60,
        "avg_forward_excess_return": 0.01,
    }
    remove = {
        "sample_count": 60,
        "information_coefficient": 0.01,
        "hit_rate": 0.49,
        "avg_forward_excess_return": -0.01,
    }
    watch = {
        "sample_count": 10,
        "information_coefficient": 0.20,
        "hit_rate": 0.70,
        "avg_forward_excess_return": 0.02,
    }

    assert classify_signal_strength(keep) == "Keep"
    assert classify_signal_strength(remove) == "Remove"
    assert classify_signal_strength(watch) == "Watch"
