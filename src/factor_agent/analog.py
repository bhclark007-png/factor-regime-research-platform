from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature matrix with stable imputation for distance work."""
    numeric = features.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    numeric = numeric.ffill().bfill()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return numeric.dropna(axis=1, how="any")


def find_historical_analogs(
    features: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n: int = 5,
    min_history: int = 36,
) -> dict:
    """Find historical environments most similar to the latest observation.

    Similarity is calculated using Euclidean distance on expanding z-scores.
    The current observation is excluded, and analog rows must have subsequent
    factor returns available.
    """
    X = _clean_feature_matrix(features)
    common_index = X.index.intersection(forward_returns.dropna(how="all").index)
    X = X.loc[common_index]
    fwd = forward_returns.loc[common_index]

    if len(X) <= min_history:
        return {"analogs": [], "summary": "Insufficient history for analog analysis."}

    latest_date = X.index[-1]
    history = X.iloc[:-1]
    latest = X.iloc[-1]

    means = history.expanding(min_periods=min_history).mean().iloc[-1]
    stds = history.expanding(min_periods=min_history).std().iloc[-1].replace(0, np.nan)
    usable = stds.dropna().index
    if len(usable) == 0:
        return {"analogs": [], "summary": "No stable features for analog analysis."}

    z_history = ((history[usable] - means[usable]) / stds[usable]).clip(-4, 4)
    z_latest = ((latest[usable] - means[usable]) / stds[usable]).clip(-4, 4)
    distances = ((z_history - z_latest) ** 2).sum(axis=1) ** 0.5
    candidates = distances.sort_values().head(n)

    analogs = []
    for date, distance in candidates.items():
        returns = fwd.loc[date].dropna().sort_values(ascending=False)
        if returns.empty:
            continue
        analogs.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "distance": float(distance),
                "best_factor": str(returns.index[0]),
                "best_factor_forward_excess_return": float(returns.iloc[0]),
                "factor_forward_excess_returns": {
                    str(factor): float(value) for factor, value in returns.items()
                },
            }
        )

    if not analogs:
        summary = "No analog periods had complete forward factor returns."
    else:
        best_counts = pd.Series([a["best_factor"] for a in analogs]).value_counts()
        top_factor = best_counts.index[0]
        summary = (
            f"{top_factor} led in {int(best_counts.iloc[0])} of "
            f"{len(analogs)} closest historical analogs."
        )

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "feature_count": int(len(usable)),
        "analogs": analogs,
        "summary": summary,
    }
