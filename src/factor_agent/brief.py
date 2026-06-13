from __future__ import annotations

from datetime import datetime
import pandas as pd


def _fmt_pct(x):
    try:
        return f"{x:.1%}"
    except Exception:
        return "n/a"


def make_daily_brief(
    probabilities: pd.Series,
    cv_accuracy: float,
    credit_score: int,
    credit_drivers: list[str],
    stability_score: int,
    risks: list[dict],
    dynamic_risks: dict | None,
    regime: str,
    analogs: dict | None,
    data_quality: dict | None,
    feature_importances: pd.Series,
) -> str:
    top_factor = probabilities.index[0] if len(probabilities) else "unknown"
    top_prob = probabilities.iloc[0] if len(probabilities) else float("nan")

    lines = []
    lines.append("# Daily Factor Regime Brief")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## BLUF")
    lines.append("")
    lines.append(f"**Current regime:** {regime}")
    lines.append(f"**Highest-probability factor:** {top_factor} ({_fmt_pct(top_prob)})")
    lines.append(f"**Credit Leadership Score:** {credit_score}/100")
    lines.append(f"**Regime Stability Score:** {stability_score}/100")
    lines.append(f"**Backtest cross-validation hit rate:** {_fmt_pct(cv_accuracy)}")
    if data_quality and data_quality.get("data_impaired"):
        lines.append("**Data quality:** impaired; confidence reduced.")
    lines.append("")
    lines.append("## Factor Probabilities")
    lines.append("")
    lines.append("| Factor | Probability |")
    lines.append("|---|---:|")
    for factor, prob in probabilities.items():
        lines.append(f"| {factor} | {_fmt_pct(prob)} |")
    lines.append("")
    lines.append("## Why This Regime Is Supported")
    lines.append("")
    if credit_drivers:
        for d in credit_drivers[:6]:
            lines.append(f"- {d}")
    else:
        lines.append(
            "- No major credit driver was detected from the latest public data."
        )
    lines.append("")
    lines.append("## Regime Change Risks")
    lines.append("")
    if dynamic_risks and dynamic_risks.get("transition_probability") is not None:
        lines.append(
            f"**Estimated transition pressure:** {_fmt_pct(dynamic_risks['transition_probability'])}"
        )
        lines.append("")
        lines.append("| Indicator | Risk | Stress Percentile | Direction |")
        lines.append("|---|---|---:|---|")
        for r in dynamic_risks.get("risks", [])[:5]:
            lines.append(
                f"| {r['indicator']} | {r['risk']} | {_fmt_pct(r['stress_percentile'])} | {r['direction']} |"
            )
        lines.append("")

    if risks:
        lines.append(
            "| Risk | Current | Warning Zone | Severity | Portfolio Implication |"
        )
        lines.append("|---|---|---|---|---|")
        for r in risks:
            lines.append(
                f"| {r['risk']} | {r['current']} | {r['warning']} | {r['severity']} | {r['implication']} |"
            )
    else:
        lines.append(
            "No major regime-change warnings were triggered. This does not mean risk is absent; it means the monitored public indicators are not currently flashing transition signals."
        )
    lines.append("")
    lines.append("## Historical Analog Summary")
    lines.append("")
    if analogs and analogs.get("analogs"):
        lines.append(analogs.get("summary", "Historical analogs are available."))
        lines.append("")
        lines.append(
            "| Date | Distance | Best Forward Factor | Forward Excess Return |"
        )
        lines.append("|---|---:|---|---:|")
        for analog in analogs.get("analogs", [])[:5]:
            lines.append(
                f"| {analog['date']} | {analog['distance']:.2f} | "
                f"{analog['best_factor']} | {_fmt_pct(analog['best_factor_forward_excess_return'])} |"
            )
    else:
        lines.append("Insufficient comparable history for a reliable analog summary.")
    lines.append("")
    lines.append("## Most Important Historical Drivers in the Model")
    lines.append("")
    lines.append(
        "These are the variables the prototype model relied on most. Treat this as a research clue, not a causal conclusion."
    )
    lines.append("")
    lines.append("| Feature | Importance |")
    lines.append("|---|---:|")
    for feature, imp in feature_importances.head(12).items():
        lines.append(f"| {feature} | {imp:.3f} |")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        f"The model currently favors **{top_factor}**. The key question for tomorrow's brief is whether credit, volatility, and growth data continue to confirm this regime or whether the risk triggers above begin to move into warning territory."
    )
    lines.append("")
    lines.append("_Prototype only. Not investment advice._")
    return "\n".join(lines)
