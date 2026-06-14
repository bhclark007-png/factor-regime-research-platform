from __future__ import annotations

import pandas as pd

from factor_agent.features import build_factor_excess_returns, forward_factor_winner


def test_factor_excess_returns_and_winner() -> None:
    index = pd.date_range("2024-01-31", periods=8, freq="ME")
    prices = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
            "MTUM": [100, 103, 106, 108, 110, 112, 115, 118],
            "QUAL": [100, 101, 103, 104, 105, 106, 107, 108],
        },
        index=index,
    )

    factor_excess = build_factor_excess_returns(prices)
    forward, winner = forward_factor_winner(factor_excess, horizon_months=3)

    assert "momentum" in factor_excess
    assert winner.dropna().iloc[0] == "momentum"
    assert len(forward) == len(factor_excess)
