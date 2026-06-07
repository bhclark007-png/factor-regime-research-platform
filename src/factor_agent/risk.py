from __future__ import annotations

import pandas as pd


MONITORED_RISK_FEATURES = {
    "hy_oas_3m_chg": "Credit spread deterioration",
    "ccc_minus_hy_3m_chg": "Lower-quality credit underperformance",
    "vix_1m_chg": "Volatility acceleration",
    "ism_3m_chg": "Manufacturing growth deterioration",
    "claims_3m_pct_chg": "Labor-market deterioration",
    "cpi_3m_ann": "Inflation reacceleration",
    "curve_2s10s_3m_chg": "Yield-curve shift",
}


def dynamic_regime_risks(features: pd.DataFrame, top_n: int = 5) -> dict:
    """Rank current regime risks using historical percentiles.

    This intentionally avoids fixed warning thresholds. Each indicator is scored
    by how unusual its latest value is versus its own history. Directional signs
    are normalized so higher percentile means higher transition concern.
    """
    rows = []
    numeric = features.select_dtypes(include=["number"]).ffill()
    if numeric.empty:
        return {"transition_probability": None, "risks": []}

    for feature, label in MONITORED_RISK_FEATURES.items():
        if feature not in numeric:
            continue
        series = numeric[feature].dropna()
        if len(series) < 36:
            continue
        latest = float(series.iloc[-1])
        if feature in {"ism_3m_chg", "curve_2s10s_3m_chg"}:
            stress_series = -series
            latest_stress = -latest
        else:
            stress_series = series
            latest_stress = latest

        percentile = float((stress_series <= latest_stress).mean())
        transition_frequency = float((stress_series >= stress_series.quantile(0.8)).mean())
        rows.append(
            {
                "indicator": feature,
                "risk": label,
                "latest_value": latest,
                "stress_percentile": percentile,
                "historical_frequency_before_transitions": transition_frequency,
                "direction": "lower is riskier" if feature in {"ism_3m_chg", "curve_2s10s_3m_chg"} else "higher is riskier",
            }
        )

    risks = sorted(rows, key=lambda row: row["stress_percentile"], reverse=True)[:top_n]
    if not risks:
        transition_probability = None
    else:
        transition_probability = float(
            min(0.95, max(0.05, sum(row["stress_percentile"] for row in risks) / len(risks)))
        )

    return {
        "transition_probability": transition_probability,
        "risks": risks,
    }
