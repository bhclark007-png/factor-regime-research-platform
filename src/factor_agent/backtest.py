from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


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


def clean_model_frame(features: pd.DataFrame, winner: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
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
    n_estimators: int = 100,
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

    for i in range(min_train_months, len(X)):
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
        "avg_forward_excess_return": float(selected.mean()) if len(predictions) else None,
        "avg_excess_vs_equal_weight_factors": float(active_vs_equal_weight.mean()) if len(predictions) else None,
        "avg_excess_vs_spy": float(selected.mean()) if len(predictions) else None,
        "information_ratio": _information_ratio(selected),
        "information_ratio_vs_equal_weight_factors": _information_ratio(active_vs_equal_weight),
        "max_drawdown": _max_drawdown(selected),
        "turnover_proxy": _turnover_proxy(predictions["selected_factor"]),
        "equal_weight_factor_avg_forward_excess_return": float(equal_weight.mean()) if len(predictions) else None,
        "spy_forward_excess_return": 0.0,
    }


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


def validate_factor_model(
    features: pd.DataFrame,
    factor_excess: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6),
    min_train_months: int = 48,
    n_estimators: int = 100,
) -> dict:
    """Compare model-selected factor exposure with equal-weight factors and SPY."""
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
        summaries.append(results[str(horizon)]["summary"])

    return {
        "version": "0.5",
        "objective": "Compare model-selected factor exposure against equal-weight factors and SPY.",
        "horizons": list(horizons),
        "summary": summaries,
        "by_horizon": results,
    }
