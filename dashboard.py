from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LATEST_JSON = APP_ROOT / "output" / "latest" / "run_result.json"
LATEST_BRIEF = APP_ROOT / "output" / "latest" / "daily_brief.md"
RUN_HISTORY = APP_ROOT / "output" / "run_history.csv"


st.set_page_config(
    page_title="Factor Regime Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      :root {
        --bg: #f7f8fb;
        --panel: #ffffff;
        --ink: #17202a;
        --muted: #5b6673;
        --line: #dde3ea;
        --good: #1f7a4d;
        --warn: #b26a00;
        --bad: #a33636;
        --accent: #22577a;
      }
      .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.4rem;
        max-width: 1280px;
      }
      h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
      [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        min-height: 108px;
      }
      [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.82rem;
      }
      [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: clamp(1.25rem, 2vw, 1.85rem);
      }
      .section {
        margin-top: 1.35rem;
      }
      .status-pill {
        display: inline-block;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.78rem;
        color: var(--muted);
        background: #fff;
        margin-right: 0.35rem;
      }
      .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--accent);
        color: #fff;
        background: var(--accent);
        min-height: 2.4rem;
      }
      .stButton > button:hover {
        border-color: #16384f;
        background: #16384f;
        color: #fff;
      }
      div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
      }
      @media (max-width: 720px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        [data-testid="stMetric"] { min-height: 92px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_result() -> dict:
    if not LATEST_JSON.exists():
        st.error("No backend result found. Run the backend once to create output/latest/run_result.json.")
        st.stop()
    return json.loads(LATEST_JSON.read_text(encoding="utf-8"))


def pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}/100"


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def run_backend(refresh_data: bool) -> tuple[int, str]:
    cmd = [sys.executable, "-u", "run_agent.py", "--output", "output"]
    if refresh_data:
        cmd.append("--refresh-data")
    proc = subprocess.run(
        cmd,
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        timeout=420,
    )
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    return proc.returncode, output


def severity_rank(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(value).lower(), 3)


def source_health_frame(result: dict) -> pd.DataFrame:
    rows = result.get("data_status", {}).get("sources", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    visible = ["name", "ticker", "source", "status", "rows", "error"]
    for col in visible:
        if col not in df:
            df[col] = None
    return df[visible].sort_values(["status", "name"])


def render_header(result: dict) -> None:
    regime = result["regime"]
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.title("Factor Regime Dashboard")
        st.caption(f"Run {result['run_id']} | Generated {result['generated_at']}")
    with right:
        st.markdown(
            f"""
            <span class="status-pill">{regime['label']}</span>
            <span class="status-pill">{regime['top_factor']}</span>
            """,
            unsafe_allow_html=True,
        )


def render_metrics(result: dict) -> None:
    regime = result["regime"]
    cols = st.columns(5)
    cols[0].metric("Current regime", regime["label"])
    cols[1].metric("Top factor", regime["top_factor"], pct(regime["top_factor_probability"]))
    cols[2].metric("Credit leadership", score(regime["credit_leadership_score"]))
    cols[3].metric("Regime stability", score(regime["regime_stability_score"]))
    cols[4].metric("CV hit rate", pct(regime["cv_accuracy"]))


def render_probabilities(result: dict) -> None:
    st.subheader("Factor Probabilities")
    probs = frame_from_records(result.get("factor_probabilities", []))
    if probs.empty:
        st.info("No probabilities available.")
        return
    probs = probs.sort_values("probability", ascending=True).set_index("name")
    st.bar_chart(probs["probability"], height=260)
    table = probs.sort_values("probability", ascending=False).reset_index()
    table["probability"] = table["probability"].map(pct)
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_risks_and_drivers(result: dict) -> None:
    left, right = st.columns([0.58, 0.42])
    with left:
        st.subheader("Regime Change Risks")
        dynamic = result.get("dynamic_risks", {})
        if dynamic.get("transition_probability") is not None:
            st.metric("Transition pressure", pct(dynamic["transition_probability"]))
            dynamic_rows = frame_from_records(dynamic.get("risks", []))
            if not dynamic_rows.empty:
                dynamic_rows["stress_percentile"] = dynamic_rows["stress_percentile"].map(pct)
                st.dataframe(dynamic_rows, hide_index=True, use_container_width=True)
        risks = frame_from_records(result.get("risks", []))
        if risks.empty:
            st.success("No active risk triggers.")
        else:
            risks["_rank"] = risks["severity"].map(severity_rank)
            risks = risks.sort_values(["_rank", "risk"]).drop(columns=["_rank"])
            st.dataframe(risks, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Credit Drivers")
        drivers = result.get("credit_drivers", [])
        if not drivers:
            st.info("No major credit driver detected.")
        for driver in drivers:
            st.markdown(f"- {driver}")


def render_model_detail(result: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Top Model Drivers")
        features = frame_from_records(result.get("feature_importances", []))
        if features.empty:
            st.info("No feature importances available.")
        else:
            features = features.head(12).copy()
            features["importance"] = features["importance"].map(lambda x: f"{x:.3f}")
            st.dataframe(features, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Backtest Summary")
        backtest = frame_from_records(result.get("backtest_summary", []))
        if backtest.empty:
            st.info("No backtest summary available.")
        else:
            for col in ["avg_forward_excess_return", "median_forward_excess_return", "win_rate_vs_spy"]:
                if col in backtest:
                    backtest[col] = backtest[col].map(pct)
            st.dataframe(backtest, hide_index=True, use_container_width=True)


def render_analogs(result: dict) -> None:
    st.subheader("Historical Analogs")
    analogs = result.get("historical_analogs", {})
    st.caption(analogs.get("summary", "No analog summary available."))
    rows = frame_from_records(analogs.get("analogs", []))
    if rows.empty:
        st.info("No analog periods available.")
        return
    display = rows[["date", "distance", "best_factor", "best_factor_forward_excess_return"]].copy()
    display["distance"] = display["distance"].map(lambda x: f"{x:.2f}")
    display["best_factor_forward_excess_return"] = display["best_factor_forward_excess_return"].map(pct)
    st.dataframe(display, hide_index=True, use_container_width=True)


def render_source_health(result: dict) -> None:
    st.subheader("Data Source Health")
    data_status = result.get("data_status", {})
    cols = st.columns(3)
    cols[0].metric("Sources", data_status.get("sources_total", 0))
    cols[1].metric("Successful", data_status.get("sources_successful", 0))
    cols[2].metric("Failed", data_status.get("sources_failed", 0))

    health = source_health_frame(result)
    if health.empty:
        st.info("No source status records available.")
    else:
        st.dataframe(health, hide_index=True, use_container_width=True)


def render_history() -> None:
    if not RUN_HISTORY.exists():
        return
    st.subheader("Run History")
    history = pd.read_csv(RUN_HISTORY).tail(20).sort_values("generated_at", ascending=False)
    for col in ["top_factor_probability", "cv_accuracy"]:
        if col in history:
            history[col] = history[col].map(pct)
    st.dataframe(history, hide_index=True, use_container_width=True)


def render_brief() -> None:
    if not LATEST_BRIEF.exists():
        return
    with st.expander("Daily Brief", expanded=False):
        st.markdown(LATEST_BRIEF.read_text(encoding="utf-8"))


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Controls")
        refresh_data = st.toggle("Force fresh data", value=False)
        if st.button("Run Backend", use_container_width=True):
            with st.spinner("Running backend..."):
                try:
                    code, output = run_backend(refresh_data=refresh_data)
                except subprocess.TimeoutExpired:
                    st.error("Backend run timed out.")
                    st.stop()
            if code == 0:
                st.success("Backend run completed.")
                with st.expander("Run log"):
                    st.code(output)
                st.rerun()
            st.error("Backend run failed.")
            st.code(output)


def main() -> None:
    render_sidebar()
    result = load_result()
    render_header(result)
    render_metrics(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_probabilities(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_risks_and_drivers(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_model_detail(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_analogs(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_source_health(result)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    render_history()
    render_brief()


if __name__ == "__main__":
    main()
