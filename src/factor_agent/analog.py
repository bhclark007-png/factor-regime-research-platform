from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_FEATURES = [
    "hy_oas",
    "hy_oas_3m_chg",
    "ccc_minus_hy_3m_chg",
    "vix",
    "vix_1m_chg",
    "ism_mfg",
    "ism_3m_chg",
    "cpi_3m_ann",
    "curve_2s10s",
]


def _clean_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature matrix with stable imputation for distance work."""
    numeric = features.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    numeric = numeric.ffill().bfill()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return numeric.dropna(axis=1, how="any")


def _regime_name(row: pd.Series) -> str:
    """Assign a simple descriptive regime label from observed macro conditions."""
    hy_change = row.get("hy_oas_3m_chg", np.nan)
    vix = row.get("vix", np.nan)
    ism = row.get("ism_mfg", np.nan)
    cpi = row.get("cpi_3m_ann", np.nan)

    if (pd.notna(hy_change) and hy_change > 50) or (pd.notna(vix) and vix > 25):
        return "Credit/Volatility Stress"
    if pd.notna(ism) and ism < 50:
        return "Growth Slowdown"
    if pd.notna(cpi) and cpi > 4:
        return "Inflation Reacceleration"
    if pd.notna(ism) and ism >= 52 and (pd.isna(hy_change) or hy_change <= 0):
        return "Risk-On Expansion"
    return "Mixed / Data-Dependent"


def _feature_weights(history: pd.DataFrame) -> pd.Series:
    """Weight features by non-missing coverage and stability."""
    coverage = history.notna().mean()
    volatility = history.std().replace(0, np.nan)
    weights = coverage / volatility.rank(pct=True).replace(0, np.nan)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(coverage)
    weights = weights / weights.mean()
    return weights.clip(0.25, 3.0)


def _forward_return_payload(
    date: pd.Timestamp,
    horizon_forward_returns: dict[int, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    payload = {}
    for horizon, returns in sorted(horizon_forward_returns.items()):
        if date not in returns.index:
            continue
        row = returns.loc[date].dropna().sort_values(ascending=False)
        payload[f"{horizon}m"] = {
            str(factor): float(value) for factor, value in row.items()
        }
    return payload


def find_historical_analogs(
    features: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon_forward_returns: dict[int, pd.DataFrame] | None = None,
    n: int = 5,
    min_history: int = 36,
) -> dict:
    """Find historical environments most similar to the latest observation.

    Similarity uses weighted Euclidean distance on z-scored macro/credit
    features. The current observation is excluded, and analog rows must have
    subsequent factor returns available.
    """
    X = _clean_feature_matrix(features)
    preferred = [col for col in REGIME_FEATURES if col in X.columns]
    if preferred:
        X = X[preferred]
    horizon_forward_returns = horizon_forward_returns or {}
    if not horizon_forward_returns:
        horizon_forward_returns = {0: forward_returns}

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

    weights = _feature_weights(history[usable])
    z_history = ((history[usable] - means[usable]) / stds[usable]).clip(-4, 4)
    z_latest = ((latest[usable] - means[usable]) / stds[usable]).clip(-4, 4)
    distances = (((z_history - z_latest) ** 2) * weights[usable]).sum(axis=1) ** 0.5
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
                "historical_regime": _regime_name(features.loc[date]),
                "similarity_score": float(1 / (1 + distance)),
                "factor_forward_excess_returns": {
                    str(factor): float(value) for factor, value in returns.items()
                },
                "forward_factor_returns_by_horizon": _forward_return_payload(
                    pd.Timestamp(date), horizon_forward_returns
                ),
            }
        )

    if not analogs:
        summary = "No analog periods had complete forward factor returns."
    else:
        best_counts = pd.Series([a["best_factor"] for a in analogs]).value_counts()
        top_factor = best_counts.index[0]
        regime_counts = pd.Series(
            [a["historical_regime"] for a in analogs]
        ).value_counts()
        summary = (
            f"{top_factor} led in {int(best_counts.iloc[0])} of "
            f"{len(analogs)} closest historical analogs; "
            f"{regime_counts.index[0]} was the most common analog regime."
        )

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "feature_count": int(len(usable)),
        "similarity_method": "weighted_zscore_euclidean_distance",
        "closest_historical_regimes": (
            pd.Series([a["historical_regime"] for a in analogs])
            .value_counts()
            .rename_axis("regime")
            .reset_index(name="count")
            .to_dict(orient="records")
            if analogs
            else []
        ),
        "analogs": analogs,
        "summary": summary,
    }
