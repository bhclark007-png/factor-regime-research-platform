"""Validation and baseline comparisons for factor leadership models."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CREDIT_VALIDATION_VARIABLES = [
    "hy_oas",
    "ig_oas",
    "ccc_oas",
    "hy_oas_1m_chg",
    "hy_oas_3m_chg",
    "ig_oas_3m_chg",
    "ccc_oas_3m_chg",
    "ccc_minus_hy_3m_chg",
    "vix",
    "vix_1m_chg",
]


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


def clean_model_frame(
    features: pd.DataFrame, winner: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare aligned model features and labels for validation."""
    dataset = features.join(winner.rename("winner")).dropna(subset=["winner"])
    X = dataset.drop(columns=["winner"]).dropna(axis=1, how="all")
    X = X.ffill().bfill()
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="any")
    y = dataset.loc[X.index, "winner"]
    return X, y


def walk_forward_factor_predictions(
    features: pd.DataFrame,
    forward_returns: pd.DataFrame,
    winner: pd.Series,
    min_train_months: int = 48,
    n_estimators: int = 20,
    validation_step_months: int = 3,
) -> pd.DataFrame:
    """Generate expanding-window factor selections for validation.

    At each month, the model trains only on observations available before that
    month and selects the factor with the highest predicted probability.
    """
    X, y = clean_model_frame(features, winner)
    common_index = X.index.intersection(forward_returns.dropna(how="all").index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    fwd = forward_returns.loc[common_index]

    rows = []
    if len(X) <= min_train_months:
        return pd.DataFrame()

    step = max(1, validation_step_months)
    for i in range(min_train_months, len(X), step):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            min_samples_leaf=6,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        probabilities = pd.Series(
            model.predict_proba(X.iloc[[i]])[0],
            index=model.classes_,
        ).sort_values(ascending=False)

        selected = str(probabilities.index[0])
        realized = fwd.iloc[i].dropna()
        if selected not in realized or realized.empty:
            continue
        actual_winner = str(realized.idxmax())
        rows.append(
            {
                "date": X.index[i],
                "selected_factor": selected,
                "actual_winner": actual_winner,
                "confidence": float(probabilities.iloc[0]),
                "selected_forward_excess_return": float(realized[selected]),
                "equal_weight_factor_excess_return": float(realized.mean()),
                "spy_forward_excess_return": 0.0,
                "hit": bool(selected == actual_winner),
            }
        )

    return pd.DataFrame(rows)


def _information_ratio(returns: pd.Series) -> float | None:
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return None
    return float(returns.mean() / std)


def _max_drawdown(returns: pd.Series) -> float | None:
    returns = returns.dropna()
    if returns.empty:
        return None
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min())


def _turnover_proxy(selected: pd.Series) -> float | None:
    selected = selected.dropna()
    if len(selected) < 2:
        return None
    return float((selected != selected.shift(1)).iloc[1:].mean())


def _strategy_summary(predictions: pd.DataFrame) -> dict:
    selected = predictions["selected_forward_excess_return"]
    equal_weight = predictions["equal_weight_factor_excess_return"]
    active_vs_equal_weight = selected - equal_weight

    return {
        "observations": int(len(predictions)),
        "hit_rate": float(predictions["hit"].mean()) if len(predictions) else None,
        "avg_forward_excess_return": (
            float(selected.mean()) if len(predictions) else None
        ),
        "avg_excess_vs_equal_weight_factors": (
            float(active_vs_equal_weight.mean()) if len(predictions) else None
        ),
        "avg_excess_vs_spy": float(selected.mean()) if len(predictions) else None,
        "information_ratio": _information_ratio(selected),
        "information_ratio_vs_equal_weight_factors": _information_ratio(
            active_vs_equal_weight
        ),
        "max_drawdown": _max_drawdown(selected),
        "turnover_proxy": _turnover_proxy(predictions["selected_factor"]),
        "equal_weight_factor_avg_forward_excess_return": (
            float(equal_weight.mean()) if len(predictions) else None
        ),
        "spy_forward_excess_return": 0.0,
    }


def _prediction_summary(
    predictions: pd.DataFrame, return_col: str = "selected_forward_excess_return"
) -> dict:
    if predictions.empty:
        return {
            "observations": 0,
            "hit_rate": None,
            "avg_forward_excess_return": None,
            "information_ratio": None,
            "max_drawdown": None,
            "turnover_proxy": None,
        }
    returns = predictions[return_col]
    return {
        "observations": int(len(predictions)),
        "hit_rate": float(predictions["hit"].mean()) if "hit" in predictions else None,
        "avg_forward_excess_return": float(returns.mean()),
        "information_ratio": _information_ratio(returns),
        "max_drawdown": _max_drawdown(returns),
        "turnover_proxy": (
            _turnover_proxy(predictions["selected_factor"])
            if "selected_factor" in predictions
            else None
        ),
    }


def _baseline_predictions(
    features: pd.DataFrame,
    factor_excess: pd.DataFrame,
    forward_returns: pd.DataFrame,
    winner: pd.Series,
    min_train_months: int,
    baseline: str,
    validation_step_months: int,
) -> pd.DataFrame:
    X, y = clean_model_frame(features, winner)
    common_index = X.index.intersection(forward_returns.dropna(how="all").index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    fwd = forward_returns.loc[common_index]
    factor_history = factor_excess.reindex(common_index)
    rows = []

    step = max(1, validation_step_months)
    for i in range(min_train_months, len(X), step):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        realized = fwd.iloc[i].dropna()
        if realized.empty:
            continue

        selected = None
        confidence = None
        if baseline == "previous_winner":
            selected = str(y_train.iloc[-1])
            confidence = 1.0
        elif baseline == "factor_momentum":
            trailing = factor_history.iloc[max(0, i - 6) : i].sum().dropna()
            if not trailing.empty:
                selected = str(trailing.idxmax())
                confidence = float(trailing.rank(pct=True).max())
        elif baseline == "logistic_regression":
            if y_train.nunique() < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, solver="lbfgs"),
            )
            model.fit(X_train, y_train)
            probabilities = pd.Series(
                model.predict_proba(X.iloc[[i]])[0], index=model.classes_
            ).sort_values(ascending=False)
            selected = str(probabilities.index[0])
            confidence = float(probabilities.iloc[0])
        elif baseline == "decision_tree":
            if y_train.nunique() < 2:
                continue
            model = DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=8, random_state=42
            )
            model.fit(X_train, y_train)
            probabilities = pd.Series(
                model.predict_proba(X.iloc[[i]])[0], index=model.classes_
            ).sort_values(ascending=False)
            selected = str(probabilities.index[0])
            confidence = float(probabilities.iloc[0])

        if selected not in realized:
            continue
        rows.append(
            {
                "date": X.index[i],
                "selected_factor": selected,
                "actual_winner": str(realized.idxmax()),
                "confidence": confidence,
                "selected_forward_excess_return": float(realized[selected]),
                "hit": bool(selected == str(realized.idxmax())),
            }
        )

    return pd.DataFrame(rows)


def _confusion_matrix_records(predictions: pd.DataFrame) -> list[dict]:
    labels = sorted(
        set(predictions["selected_factor"].dropna().astype(str))
        | set(predictions["actual_winner"].dropna().astype(str))
    )
    if not labels:
        return []
    matrix = confusion_matrix(
        predictions["actual_winner"],
        predictions["selected_factor"],
        labels=labels,
    )
    rows = []
    for actual, row in zip(labels, matrix):
        for predicted, count in zip(labels, row):
            rows.append({"actual": actual, "predicted": predicted, "count": int(count)})
    return rows


def validate_credit_leadership(
    features: pd.DataFrame,
    winner: pd.Series,
    variables: list[str] | None = None,
) -> dict:
    """Estimate which credit/volatility variables explain future factor leadership.

    This is intentionally simple and interpretable: each variable is scored by
    the spread between factor-specific means and by absolute correlation with
    one-vs-rest future factor leadership indicators.
    """
    variables = variables or CREDIT_VALIDATION_VARIABLES
    dataset = features.join(winner.rename("winner")).dropna(subset=["winner"])
    if dataset.empty:
        return {"method": "mean_spread_and_one_vs_rest_correlation", "variables": []}

    rows = []
    for variable in variables:
        if variable not in dataset:
            continue
        series = pd.to_numeric(dataset[variable], errors="coerce")
        usable = dataset.assign(_variable=series).dropna(subset=["_variable"])
        if len(usable) < 24 or usable["winner"].nunique() < 2:
            continue

        factor_means = usable.groupby("winner")["_variable"].mean().sort_values()
        mean_spread = float(factor_means.max() - factor_means.min())
        correlations = {}
        for factor in sorted(usable["winner"].astype(str).unique()):
            target = (usable["winner"].astype(str) == factor).astype(float)
            corr = usable["_variable"].corr(target)
            if pd.notna(corr):
                correlations[factor] = float(corr)

        max_abs_correlation = (
            max(abs(value) for value in correlations.values()) if correlations else 0.0
        )
        explanatory_value = float(max_abs_correlation * mean_spread)
        rows.append(
            {
                "variable": variable,
                "observations": int(len(usable)),
                "mean_spread_between_future_winners": mean_spread,
                "max_abs_one_vs_rest_correlation": float(max_abs_correlation),
                "explanatory_value": explanatory_value,
                "factor_mean_values": {
                    str(factor): float(value) for factor, value in factor_means.items()
                },
                "one_vs_rest_correlations": correlations,
            }
        )

    rows = sorted(rows, key=lambda row: row["explanatory_value"], reverse=True)
    return {
        "method": "mean_spread_and_one_vs_rest_correlation",
        "objective": "Evaluate whether credit spreads, spread changes, and volatility help explain future factor leadership.",
        "variables": rows,
        "top_variables": rows[:5],
    }


def validate_factor_model(
    features: pd.DataFrame,
    factor_excess: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6),
    min_train_months: int = 48,
    n_estimators: int = 20,
    max_validation_months: int = 120,
    validation_step_months: int = 3,
) -> dict:
    """Compare model-selected factor exposure with equal-weight factors and SPY."""
    factor_excess = factor_excess.dropna(how="all").sort_index()
    if max_validation_months and len(factor_excess) > max_validation_months:
        validation_start = factor_excess.index[-max_validation_months]
        factor_excess = factor_excess.loc[factor_excess.index >= validation_start]
        features = features.loc[features.index >= validation_start]
    else:
        validation_start = (
            factor_excess.index.min() if not factor_excess.empty else None
        )

    results = {}
    summaries = []

    for horizon in horizons:
        forward = factor_excess.rolling(horizon).sum().shift(-horizon)
        winner = forward.dropna(how="all").idxmax(axis=1).rename("winner")
        predictions = walk_forward_factor_predictions(
            features=features,
            forward_returns=forward,
            winner=winner,
            min_train_months=min_train_months,
            n_estimators=n_estimators,
            validation_step_months=validation_step_months,
        )
        if predictions.empty:
            summary = {
                "horizon_months": horizon,
                "observations": 0,
                "hit_rate": None,
                "avg_forward_excess_return": None,
                "information_ratio": None,
                "max_drawdown": None,
                "turnover_proxy": None,
            }
            results[str(horizon)] = {
                "summary": summary,
                "confusion_matrix": [],
                "predictions": [],
            }
        else:
            summary = {"horizon_months": horizon, **_strategy_summary(predictions)}
            records = predictions.copy()
            records["date"] = records["date"].dt.strftime("%Y-%m-%d")
            results[str(horizon)] = {
                "summary": summary,
                "confusion_matrix": _confusion_matrix_records(predictions),
                "predictions": records.to_dict(orient="records"),
            }

        baseline_results = {}
        for baseline in [
            "previous_winner",
            "factor_momentum",
            "logistic_regression",
            "decision_tree",
        ]:
            baseline_predictions = _baseline_predictions(
                features,
                factor_excess,
                forward,
                winner,
                min_train_months,
                baseline,
                validation_step_months,
            )
            baseline_results[baseline] = _prediction_summary(baseline_predictions)

        equal_weight_returns = (
            forward.loc[predictions["date"]].mean(axis=1)
            if not predictions.empty
            else pd.Series(dtype=float)
        )
        baseline_results["equal_weight_factors"] = {
            "observations": int(len(equal_weight_returns)),
            "hit_rate": None,
            "avg_forward_excess_return": (
                float(equal_weight_returns.mean())
                if len(equal_weight_returns)
                else None
            ),
            "information_ratio": _information_ratio(equal_weight_returns),
            "max_drawdown": _max_drawdown(equal_weight_returns),
            "turnover_proxy": 0.0,
        }
        baseline_results["spy"] = {
            "observations": int(len(equal_weight_returns)),
            "hit_rate": None,
            "avg_forward_excess_return": 0.0,
            "information_ratio": None,
            "max_drawdown": 0.0,
            "turnover_proxy": 0.0,
        }

        model_avg = results[str(horizon)]["summary"].get("avg_forward_excess_return")
        comparisons = []
        for name, baseline_summary in baseline_results.items():
            baseline_avg = baseline_summary.get("avg_forward_excess_return")
            adds_value = None
            excess = None
            if model_avg is not None and baseline_avg is not None:
                excess = float(model_avg - baseline_avg)
                adds_value = bool(excess > 0)
            comparisons.append(
                {
                    "baseline": name,
                    "model_excess_return_advantage": excess,
                    "model_adds_value": adds_value,
                }
            )

        results[str(horizon)]["baselines"] = baseline_results
        results[str(horizon)]["model_value_add"] = comparisons
        summaries.append(results[str(horizon)]["summary"])

    credit_validation = validate_credit_leadership(
        features,
        factor_excess.rolling(3).sum().shift(-3).dropna(how="all").idxmax(axis=1),
    )

    return {
        "version": "0.6.1",
        "objective": "Compare model-selected factor exposure against Random Forest, simple interpretable baselines, equal-weight factors, and SPY.",
        "model": "random_forest",
        "comparison_universe": [
            "SPY",
            "equal_weight_factors",
            "prior_month_winner",
            "factor_momentum",
            "random_forest_model",
            "simple_interpretable_decision_tree",
        ],
        "horizons": list(horizons),
        "validation_window": {
            "max_months": max_validation_months,
            "step_months": validation_step_months,
            "start": (
                validation_start.strftime("%Y-%m-%d")
                if validation_start is not None
                else None
            ),
            "end": (
                factor_excess.index.max().strftime("%Y-%m-%d")
                if not factor_excess.empty
                else None
            ),
        },
        "summary": summaries,
        "by_horizon": results,
        "credit_leadership_validation": credit_validation,
    }
