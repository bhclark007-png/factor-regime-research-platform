from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_score(x: float) -> int:
    return int(max(0, min(100, round(x))))


def credit_leadership_score(latest: pd.Series) -> tuple[int, list[str]]:
    score = 50.0
    drivers = []

    hy = latest.get("hy_oas", np.nan)
    hy_1m = latest.get("hy_oas_1m_chg", np.nan)
    hy_3m = latest.get("hy_oas_3m_chg", np.nan)
    ccc_3m = latest.get("ccc_oas_3m_chg", np.nan)
    ccc_minus_hy_3m = latest.get("ccc_minus_hy_3m_chg", np.nan)

    if pd.notna(hy):
        if hy < 350:
            score += 12
            drivers.append("HY spreads are relatively contained.")
        elif hy > 500:
            score -= 18
            drivers.append("HY spreads are stress-level elevated.")
        else:
            drivers.append("HY spreads are neither benign nor stressed.")

    if pd.notna(hy_1m):
        if hy_1m < -15:
            score += 12
            drivers.append("HY spreads tightened materially over the past month.")
        elif hy_1m > 25:
            score -= 15
            drivers.append("HY spreads widened materially over the past month.")

    if pd.notna(hy_3m):
        if hy_3m < -35:
            score += 12
            drivers.append("Three-month credit trend is improving.")
        elif hy_3m > 50:
            score -= 18
            drivers.append("Three-month credit trend is deteriorating.")

    if pd.notna(ccc_3m):
        if ccc_3m < -75:
            score += 8
            drivers.append(
                "CCC spreads are tightening, indicating improving risk appetite."
            )
        elif ccc_3m > 100:
            score -= 12
            drivers.append(
                "CCC spreads are widening, indicating rising default concern."
            )

    if pd.notna(ccc_minus_hy_3m):
        if ccc_minus_hy_3m > 50:
            score -= 10
            drivers.append("Lower-quality credit is underperforming broader HY.")
        elif ccc_minus_hy_3m < -50:
            score += 7
            drivers.append("Lower-quality credit is participating in the rally.")

    return _clip_score(score), drivers


def regime_stability_score(latest: pd.Series) -> tuple[int, list[dict]]:
    risks = []
    score = 100.0

    def add(name, current, warning, severity, implication):
        nonlocal score
        risks.append(
            {
                "risk": name,
                "current": current,
                "warning": warning,
                "severity": severity,
                "implication": implication,
            }
        )
        score -= {"low": 5, "medium": 12, "high": 22}.get(severity, 8)

    hy_1m = latest.get("hy_oas_1m_chg", np.nan)
    hy_3m = latest.get("hy_oas_3m_chg", np.nan)
    vix = latest.get("vix", np.nan)
    vix_1m = latest.get("vix_1m_chg", np.nan)
    ism = latest.get("ism_mfg", np.nan)
    ism_3m = latest.get("ism_3m_chg", np.nan)
    claims_3m = latest.get("claims_3m_pct_chg", np.nan)
    cpi_3m = latest.get("cpi_3m_ann", np.nan)

    if pd.notna(hy_1m) and hy_1m > 25:
        add(
            "Credit deterioration",
            f"HY OAS 1M change {hy_1m:.0f} bps",
            "> +25 bps/month",
            "high",
            "Favors Quality/Low Vol over Momentum/Small Cap.",
        )
    elif pd.notna(hy_3m) and hy_3m > 50:
        add(
            "Credit deterioration",
            f"HY OAS 3M change {hy_3m:.0f} bps",
            "> +50 bps/3 months",
            "high",
            "Risk-off transition probability rises.",
        )

    if pd.notna(vix) and vix > 25:
        add(
            "Volatility shock",
            f"VIX {vix:.1f}",
            "> 25",
            "high",
            "Momentum becomes vulnerable; Low Vol improves.",
        )
    elif pd.notna(vix_1m) and vix_1m > 6:
        add(
            "Volatility shock",
            f"VIX 1M change {vix_1m:.1f}",
            "> +6 points/month",
            "medium",
            "Risk appetite is weakening.",
        )

    if pd.notna(ism) and ism < 50:
        add(
            "Growth slowdown",
            f"ISM {ism:.1f}",
            "< 50",
            "medium",
            "Favors Quality and defensives; hurts Value/Small Cap.",
        )
    elif pd.notna(ism_3m) and ism_3m < -3:
        add(
            "Growth slowdown",
            f"ISM 3M change {ism_3m:.1f}",
            "< -3 points/3 months",
            "medium",
            "Cyclical factor leadership may fade.",
        )

    if pd.notna(claims_3m) and claims_3m > 0.15:
        add(
            "Labor deterioration",
            f"Claims 3M change {claims_3m:.1%}",
            "> +15%/3 months",
            "medium",
            "Recession risk rising; Quality/Low Vol favored.",
        )

    if pd.notna(cpi_3m) and cpi_3m > 4.0:
        add(
            "Inflation reacceleration",
            f"3M annualized CPI {cpi_3m:.1f}%",
            "> 4%",
            "medium",
            "Rates pressure may hurt long-duration Growth/Quality.",
        )

    return _clip_score(score), risks


def regime_label(probabilities: pd.Series, credit_score: int, stability: int) -> str:
    top = probabilities.index[0] if len(probabilities) else "unknown"
    if stability < 45:
        return "Transition / Risk-Off Watch"
    if credit_score >= 70 and top in {"momentum", "small_cap", "value"}:
        return "Risk-On Expansion"
    if top in {"quality", "low_vol"}:
        return "Defensive / Slowdown"
    if top == "value":
        return "Reflation / Cyclical"
    return "Mixed / Data-Dependent"
