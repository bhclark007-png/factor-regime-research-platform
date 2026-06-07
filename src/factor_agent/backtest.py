from __future__ import annotations

import pandas as pd


def factor_backtest_metrics(forward_returns: pd.DataFrame, winner: pd.Series) -> dict:
    """Calculate interpretable factor prediction baseline metrics.

    The current model emits the latest probability distribution rather than a
    full walk-forward probability panel. This function provides reproducible
    realized factor-return metrics that can be compared against future model
    prediction histories.
    """
    common = forward_returns.join(winner.rename("winner")).dropna()
    if common.empty:
        return {"observations": 0, "factor_metrics": []}

    factor_metrics = []
    for factor in forward_returns.columns:
        returns = common[factor].dropna()
        if returns.empty:
            continue
        cumulative = (1 + returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        factor_metrics.append(
            {
                "factor": factor,
                "observations": int(len(returns)),
                "avg_forward_excess_return": float(returns.mean()),
                "median_forward_excess_return": float(returns.median()),
                "precision_win_rate": float((returns > 0).mean()),
                "times_best_factor": int((common["winner"] == factor).sum()),
                "max_drawdown": float(drawdown.min()),
            }
        )

    return {
        "observations": int(len(common)),
        "factor_metrics": factor_metrics,
    }
