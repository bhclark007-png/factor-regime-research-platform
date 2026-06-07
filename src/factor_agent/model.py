from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


def train_factor_model(features: pd.DataFrame, winner: pd.Series):
    dataset = features.join(winner).dropna(subset=["winner"])
    if len(dataset) < 36:
        raise RuntimeError(
            f"Only {len(dataset)} usable monthly observations. Need at least ~36. "
            "ETF history may be too short or downloads may have failed."
        )

    X = dataset.drop(columns=["winner"]).dropna(axis=1, how="all")
    y = dataset["winner"]

    # Keep rows when individual macro series have publication lags or transient
    # download gaps. Random forests need a complete feature matrix.
    X = X.ffill().bfill()
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="any")

    if X.empty:
        raise RuntimeError("No usable model features remained after cleaning missing macro data.")

    n_splits = min(5, max(2, len(dataset) // 24))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in splitter.split(X):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=6,
            random_state=42,
            class_weight="balanced_subsample",
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], pred))

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=4,
        min_samples_leaf=6,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)

    latest = X.iloc[[-1]]
    probs = pd.Series(model.predict_proba(latest)[0], index=model.classes_).sort_values(ascending=False)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "model": model,
        "X": X,
        "y": y,
        "dataset": dataset,
        "cv_accuracy": float(np.mean(scores)),
        "cv_scores": scores,
        "latest_probabilities": probs,
        "feature_importances": importances,
    }


def summarize_forward_returns(forward_returns: pd.DataFrame, winner: pd.Series) -> pd.DataFrame:
    common = forward_returns.join(winner).dropna()
    rows = []
    for factor in forward_returns.columns:
        rows.append({
            "factor": factor,
            "avg_forward_excess_return": common[factor].mean(),
            "median_forward_excess_return": common[factor].median(),
            "win_rate_vs_spy": (common[factor] > 0).mean(),
            "times_best_factor": (common["winner"] == factor).sum(),
        })
    return pd.DataFrame(rows).sort_values("avg_forward_excess_return", ascending=False)
