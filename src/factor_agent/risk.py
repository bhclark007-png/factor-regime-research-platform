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


def _stress_series(features: pd.DataFrame, feature: str) -> pd.Series:
    series = features[feature].dropna()
    if feature in {"ism_3m_chg", "curve_2s10s_3m_chg"}:
        return -series
    return series


def identify_regime_breaks(
    winner: pd.Series,
    features: pd.DataFrame,
    min_streak: int = 3,
) -> list[pd.Timestamp]:
    """Define historical regime breaks from realized leadership and stress.

    A break occurs when realized factor leadership changes after a minimum
    streak and coincides with elevated credit/volatility stress when available.
    """
    aligned = winner.dropna()
    if aligned.empty:
        return []
    stress_cols = [c for c in ["hy_oas_3m_chg", "vix_1m_chg", "ccc_minus_hy_3m_chg"] if c in features]
    stress = pd.Series(0.0, index=features.index)
    for col in stress_cols:
        s = _stress_series(features, col)
        percentile = s.rank(pct=True)
        stress = stress.add(percentile.reindex(stress.index).fillna(0), fill_value=0)
    if stress_cols:
        stress = stress / len(stress_cols)

    breaks = []
    current = aligned.iloc[0]
    streak = 1
    for i in range(1, len(aligned)):
        date = aligned.index[i]
        previous = aligned.iloc[i - 1]
        value = aligned.iloc[i]
        if value == previous:
            streak += 1
            continue
        if streak >= min_streak:
            stress_value = stress.reindex([date]).iloc[0] if date in stress.index else 0
            if not stress_cols or stress_value >= 0.6:
                breaks.append(pd.Timestamp(date))
        current = value
        streak = 1
    return breaks


def regime_break_risk_monitor(
    features: pd.DataFrame,
    winner: pd.Series,
    active_regime: str,
    top_n: int = 5,
    lookback_months: int = 3,
) -> dict:
    """Rank indicators by frequency of stress before historical regime breaks."""
    numeric = features.select_dtypes(include=["number"]).ffill()
    breaks = identify_regime_breaks(winner, numeric)
    monitored = [feature for feature in MONITORED_RISK_FEATURES if feature in numeric]
    rows = []

    for feature in monitored:
        stress = _stress_series(numeric, feature).dropna()
        if len(stress) < 48:
            continue
        threshold = float(stress.quantile(0.8))
        latest_stress = float(stress.iloc[-1])
        severity = float((stress <= latest_stress).mean())
        distance = float(threshold - latest_stress)

        preceded = 0
        eligible_breaks = 0
        for break_date in breaks:
            window = stress.loc[
                (stress.index < break_date)
                & (stress.index >= break_date - pd.DateOffset(months=lookback_months))
            ]
            if window.empty:
                continue
            eligible_breaks += 1
            if (window >= threshold).any():
                preceded += 1

        frequency = float(preceded / eligible_breaks) if eligible_breaks else None
        confidence = "low"
        if eligible_breaks >= 10:
            confidence = "high"
        elif eligible_breaks >= 5:
            confidence = "medium"

        rows.append(
            {
                "indicator": feature,
                "risk": MONITORED_RISK_FEATURES[feature],
                "active_regime": active_regime,
                "historical_frequency_before_transitions": frequency,
                "current_distance_to_risk": distance,
                "severity_percentile": severity,
                "confidence": confidence,
                "break_observations": int(eligible_breaks),
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (
            -1 if row["historical_frequency_before_transitions"] is None else -row["historical_frequency_before_transitions"],
            -row["severity_percentile"],
        ),
    )[:top_n]

    transition_probability = None
    usable = [r for r in rows if r["historical_frequency_before_transitions"] is not None]
    if usable:
        transition_probability = float(
            min(
                0.95,
                sum(r["historical_frequency_before_transitions"] * r["severity_percentile"] for r in usable)
                / len(usable),
            )
        )

    return {
        "active_regime": active_regime,
        "defined_break_count": int(len(breaks)),
        "transition_probability": transition_probability,
        "top_regime_change_risks": rows,
    }
