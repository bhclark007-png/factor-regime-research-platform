from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 3, 6)
SUMMARY_COLUMNS = [
    "feature",
    "factor",
    "horizon",
    "information_coefficient",
    "hit_rate",
    "avg_forward_excess_return",
    "median_forward_excess_return",
    "win_loss_ratio",
    "sample_count",
    "recommendation",
]
DETAIL_COLUMNS = [
    "date",
    "feature",
    "factor",
    "horizon",
    "feature_value",
    "forward_excess_return",
    "signal_direction",
    "outcome_direction",
    "correct_direction",
]


def _aligned_pair(feature_series: pd.Series, target_series: pd.Series) -> pd.DataFrame:
    """Return a two-column frame with finite aligned feature and target values."""
    aligned = pd.concat(
        [feature_series.rename("feature"), target_series.rename("target")], axis=1
    ).replace([np.inf, -np.inf], np.nan)
    return aligned.dropna()


def _signal_direction(values: pd.Series) -> pd.Series:
    """
    Convert feature values into directional signals.

    The current convention is cross-time directional: observations above the feature's
    historical median are positive signals, observations below the median are negative
    signals. This keeps the function generic across level and change features while
    avoiding look-ahead because the feature value itself is observed at time t.
    """
    if values.empty:
        return pd.Series(dtype="float64", index=values.index)
    centered = values - values.median()
    return np.sign(centered).replace(0, np.nan)


def calculate_information_coefficient(
    feature_series: pd.Series, target_series: pd.Series
) -> float:
    """Calculate a Spearman rank information coefficient for one signal/target pair."""
    aligned = _aligned_pair(feature_series, target_series)
    if len(aligned) < 3:
        return float("nan")
    if aligned["feature"].nunique() <= 1 or aligned["target"].nunique() <= 1:
        return float("nan")
    value = aligned["feature"].corr(aligned["target"], method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def calculate_hit_rate(feature_series: pd.Series, target_series: pd.Series) -> float:
    """
    Calculate directional hit rate for one signal/target pair.

    Signal direction is positive when the feature is above its historical median and
    negative when below. Outcome direction is positive when the forward excess return
    is above zero and negative when below. Zero/median-tie observations are excluded.
    """
    aligned = _aligned_pair(feature_series, target_series)
    if aligned.empty:
        return float("nan")
    signals = _signal_direction(aligned["feature"])
    outcomes = np.sign(aligned["target"]).replace(0, np.nan)
    hits = pd.concat([signals.rename("signal"), outcomes.rename("outcome")], axis=1)
    hits = hits.dropna()
    if hits.empty:
        return float("nan")
    return float((hits["signal"] == hits["outcome"]).mean())


def calculate_forward_excess_returns(
    factor_returns: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[int, pd.DataFrame]:
    """
    Build forward factor excess returns for each horizon without look-ahead.

    Input rows are returns observed at time t. Output row t contains the cumulative
    return over the next horizon months, aligned to features observed at time t.
    If benchmark_returns is supplied, factor returns are first converted into excess
    returns versus that benchmark. If factor_returns are already excess returns, leave
    benchmark_returns as None.
    """
    returns = factor_returns.copy()
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] != 1:
                raise ValueError("benchmark_returns DataFrame must have one column")
            benchmark = benchmark_returns.iloc[:, 0]
        else:
            benchmark = benchmark_returns
        returns = returns.sub(benchmark, axis=0)

    outputs: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive integers")
        outputs[int(horizon)] = returns.rolling(int(horizon)).sum().shift(-int(horizon))
    return outputs


def _win_loss_ratio(values: pd.Series) -> float:
    wins = values[values > 0]
    losses = values[values < 0]
    if losses.empty:
        return float("inf") if not wins.empty else float("nan")
    if wins.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def classify_signal_strength(row: pd.Series | dict) -> str:
    """Classify a signal as Keep, Watch, or Remove using transparent thresholds."""
    sample_count = float(row.get("sample_count", 0) or 0)
    ic = row.get("information_coefficient", np.nan)
    hit_rate = row.get("hit_rate", np.nan)
    avg_return = row.get("avg_forward_excess_return", np.nan)

    if sample_count < 24:
        return "Watch"
    ic_abs = abs(float(ic)) if pd.notna(ic) else 0.0
    hit = float(hit_rate) if pd.notna(hit_rate) else 0.0
    avg = float(avg_return) if pd.notna(avg_return) else 0.0

    if ic_abs >= 0.10 and hit >= 0.55 and avg > 0:
        return "Keep"
    if ic_abs < 0.03 and hit < 0.51 and avg <= 0:
        return "Remove"
    return "Watch"


def validate_single_signal(
    feature_name: str,
    feature_series: pd.Series,
    forward_returns: pd.DataFrame | dict[int, pd.DataFrame],
    horizons: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one feature against factor forward excess returns."""
    if isinstance(forward_returns, dict):
        horizon_map = forward_returns
    else:
        if horizons is None:
            horizons = (1,)
        horizon_map = {int(horizon): forward_returns for horizon in horizons}

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    for horizon, horizon_returns in horizon_map.items():
        for factor in horizon_returns.columns:
            aligned = _aligned_pair(feature_series, horizon_returns[factor])
            if aligned.empty:
                summary_rows.append(
                    {
                        "feature": feature_name,
                        "factor": factor,
                        "horizon": int(horizon),
                        "information_coefficient": np.nan,
                        "hit_rate": np.nan,
                        "avg_forward_excess_return": np.nan,
                        "median_forward_excess_return": np.nan,
                        "win_loss_ratio": np.nan,
                        "sample_count": 0,
                    }
                )
                continue

            signal_direction = _signal_direction(aligned["feature"])
            outcome_direction = np.sign(aligned["target"]).replace(0, np.nan)
            correct_direction = signal_direction == outcome_direction
            valid_direction = signal_direction.notna() & outcome_direction.notna()

            for date, values in aligned.iterrows():
                sig = signal_direction.loc[date]
                out = outcome_direction.loc[date]
                detail_rows.append(
                    {
                        "date": date,
                        "feature": feature_name,
                        "factor": factor,
                        "horizon": int(horizon),
                        "feature_value": float(values["feature"]),
                        "forward_excess_return": float(values["target"]),
                        "signal_direction": sig if pd.notna(sig) else np.nan,
                        "outcome_direction": out if pd.notna(out) else np.nan,
                        "correct_direction": bool(correct_direction.loc[date])
                        if valid_direction.loc[date]
                        else np.nan,
                    }
                )

            summary_row = {
                "feature": feature_name,
                "factor": factor,
                "horizon": int(horizon),
                "information_coefficient": calculate_information_coefficient(
                    aligned["feature"], aligned["target"]
                ),
                "hit_rate": calculate_hit_rate(aligned["feature"], aligned["target"]),
                "avg_forward_excess_return": float(aligned["target"].mean()),
                "median_forward_excess_return": float(aligned["target"].median()),
                "win_loss_ratio": _win_loss_ratio(aligned["target"]),
                "sample_count": int(len(aligned)),
            }
            summary_row["recommendation"] = classify_signal_strength(summary_row)
            summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty and "recommendation" not in summary:
        summary["recommendation"] = summary.apply(classify_signal_strength, axis=1)
    detail = pd.DataFrame(detail_rows)
    return summary.reindex(columns=SUMMARY_COLUMNS), detail.reindex(columns=DETAIL_COLUMNS)


def validate_all_signals(
    features: pd.DataFrame,
    factor_forward_returns: dict[int, pd.DataFrame],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every feature against every factor/horizon forward return."""
    horizon_set = {int(horizon) for horizon in horizons}
    selected_returns = {
        horizon: returns
        for horizon, returns in factor_forward_returns.items()
        if int(horizon) in horizon_set
    }
    if not selected_returns:
        raise ValueError("No forward returns available for requested horizons")

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    numeric_features = features.select_dtypes(include=[np.number])
    for feature_name in numeric_features.columns:
        summary, detail = validate_single_signal(
            feature_name,
            numeric_features[feature_name],
            selected_returns,
        )
        summary_frames.append(summary)
        detail_frames.append(detail)

    summary_df = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )
    detail_df = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame(columns=DETAIL_COLUMNS)
    )
    return summary_df.reindex(columns=SUMMARY_COLUMNS), detail_df.reindex(
        columns=DETAIL_COLUMNS
    )


def signal_validation_snapshot(
    summary: pd.DataFrame, top_n: int = 10
) -> dict[str, list[dict] | bool]:
    """Create a compact run_result payload from the full signal-validation summary."""
    if summary.empty:
        return {
            "signal_validation_available": False,
            "top_signals": [],
            "weak_signals": [],
        }

    ranked = summary.copy()
    ranked["abs_information_coefficient"] = ranked["information_coefficient"].abs()
    top = ranked.sort_values(
        ["recommendation", "abs_information_coefficient", "hit_rate"],
        ascending=[True, False, False],
    ).head(top_n)
    weak = ranked.sort_values(
        ["recommendation", "information_coefficient", "hit_rate"],
        ascending=[False, True, True],
    ).head(top_n)

    keep_cols = [
        "feature",
        "factor",
        "horizon",
        "information_coefficient",
        "hit_rate",
        "avg_forward_excess_return",
        "sample_count",
        "recommendation",
    ]
    return {
        "signal_validation_available": True,
        "top_signals": top[keep_cols].to_dict(orient="records"),
        "weak_signals": weak[keep_cols].to_dict(orient="records"),
    }
