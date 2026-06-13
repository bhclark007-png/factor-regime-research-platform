from __future__ import annotations

from datetime import datetime

import pandas as pd

CRITICAL_SERIES = {
    "hy_oas": 10,
    "ig_oas": 10,
    "ccc_oas": 15,
    "vix": 7,
    "SPY": 7,
    "MTUM": 10,
    "QUAL": 10,
    "VLUE": 10,
    "USMV": 10,
    "IWM": 10,
}


def evaluate_data_quality(
    source_statuses: list[dict], as_of: str | None = None
) -> dict:
    """Evaluate failed/stale critical series and derive a confidence haircut."""
    now = pd.Timestamp(as_of or datetime.now().date())
    issues = []
    critical_failed = 0
    critical_stale = 0

    by_name = {s.get("name"): s for s in source_statuses}
    by_ticker = {s.get("ticker"): s for s in source_statuses}

    for name, stale_days in CRITICAL_SERIES.items():
        status = by_name.get(name) or by_ticker.get(name)
        if not status:
            critical_failed += 1
            issues.append(
                {"series": name, "issue": "missing_status", "severity": "high"}
            )
            continue
        if status.get("status") == "failed":
            critical_failed += 1
            issues.append(
                {
                    "series": name,
                    "issue": "failed",
                    "severity": "high",
                    "error": status.get("error"),
                }
            )
            continue
        latest = status.get("latest_observation")
        if latest:
            age_days = int((now.normalize() - pd.Timestamp(latest).normalize()).days)
            if age_days > stale_days:
                critical_stale += 1
                issues.append(
                    {
                        "series": name,
                        "issue": "stale",
                        "severity": "medium" if age_days <= stale_days * 2 else "high",
                        "latest_observation": latest,
                        "age_days": age_days,
                        "stale_after_days": stale_days,
                    }
                )

    total_critical = len(CRITICAL_SERIES)
    impairment_score = min(1.0, (critical_failed * 0.25 + critical_stale * 0.12))
    confidence_multiplier = max(0.35, 1.0 - impairment_score)
    data_impaired = critical_failed > 0 or critical_stale >= 3

    return {
        "data_impaired": data_impaired,
        "critical_failed": critical_failed,
        "critical_stale": critical_stale,
        "critical_total": total_critical,
        "confidence_multiplier": float(confidence_multiplier),
        "issues": issues,
        "summary": (
            "Data impaired: critical source failures or stale observations detected."
            if data_impaired
            else "Data quality acceptable."
        ),
    }
