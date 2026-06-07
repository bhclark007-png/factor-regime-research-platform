from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd

from .config import FRED_SERIES, ETF_TICKERS
from .data import get_fred, get_etf_prices
from .features import build_monthly_features, build_factor_excess_returns, forward_factor_winner
from .model import train_factor_model, summarize_forward_returns
from .scores import credit_leadership_score, regime_stability_score, regime_label
from .brief import make_daily_brief
from .analog import find_historical_analogs
from .backtest import factor_backtest_metrics, validate_factor_model
from .risk import dynamic_regime_risks


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _series_to_records(series: pd.Series, value_name: str) -> list[dict]:
    return [
        {"name": str(index), value_name: float(value)}
        for index, value in series.items()
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_latest(run_dir: Path, latest_dir: Path, root_dir: Path, filenames: list[str]) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, latest_dir / filename)
            shutil.copy2(source, root_dir / filename)


def _append_run_history(path: Path, summary: dict) -> None:
    row = pd.DataFrame([summary])
    if path.exists():
        history = pd.read_csv(path)
        history = pd.concat([history, row], ignore_index=True)
    else:
        history = row
    history.to_csv(path, index=False)


def run(
    start: str,
    end: str | None,
    horizon: int,
    output_dir: str,
    refresh_data: bool = False,
    run_id: str | None = None,
) -> dict:
    root = Path(output_dir)
    run_id = run_id or _timestamp_id()
    run_dir = root / "runs" / run_id
    latest_dir = root / "latest"
    cache_dir = root / ".cache" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Pulling FRED macro/market data...")
    macro, fred_status = get_fred(
        FRED_SERIES,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh_data,
        return_status=True,
    )

    print("Pulling ETF price data...")
    prices, etf_status = get_etf_prices(
        ETF_TICKERS,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh_data,
        return_status=True,
    )

    print("Building features and factor returns...")
    features = build_monthly_features(macro)
    factor_excess = build_factor_excess_returns(prices)
    forward_returns, winner = forward_factor_winner(factor_excess, horizon_months=horizon)

    print("Training regime model...")
    result = train_factor_model(features, winner)

    latest_features = result["X"].iloc[-1]
    credit_score, credit_drivers = credit_leadership_score(latest_features)
    stability_score, risks = regime_stability_score(latest_features)
    regime = regime_label(result["latest_probabilities"], credit_score, stability_score)
    analogs = find_historical_analogs(result["X"], forward_returns)
    dynamic_risks = dynamic_regime_risks(result["X"])

    print("Writing outputs...")
    probs = result["latest_probabilities"].rename("probability").to_frame()
    probs.to_csv(run_dir / "factor_probabilities.csv")

    scores = pd.DataFrame([
        {"score": "credit_leadership", "value": credit_score},
        {"score": "regime_stability", "value": stability_score},
        {"score": "cv_accuracy", "value": result["cv_accuracy"]},
    ])
    scores.to_csv(run_dir / "regime_scores.csv", index=False)

    backtest_summary = summarize_forward_returns(forward_returns, winner)
    backtest_summary.to_csv(run_dir / "backtest_summary.csv", index=False)
    backtest_metrics = factor_backtest_metrics(forward_returns, winner)
    validation = validate_factor_model(features, factor_excess)

    result["feature_importances"].rename("importance").to_csv(run_dir / "feature_importances.csv")

    brief = make_daily_brief(
        probabilities=result["latest_probabilities"],
        cv_accuracy=result["cv_accuracy"],
        credit_score=credit_score,
        credit_drivers=credit_drivers,
        stability_score=stability_score,
        risks=risks,
        dynamic_risks=dynamic_risks,
        regime=regime,
        analogs=analogs,
        feature_importances=result["feature_importances"],
    )
    (run_dir / "daily_brief.md").write_text(brief, encoding="utf-8")

    source_statuses = [s.to_dict() for s in [*fred_status, *etf_status]]
    successful_sources = [s for s in source_statuses if s["status"] != "failed"]
    failed_sources = [s for s in source_statuses if s["status"] == "failed"]

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "start": start,
            "end": end,
            "horizon_months": horizon,
            "refresh_data": refresh_data,
        },
        "regime": {
            "label": regime,
            "top_factor": str(result["latest_probabilities"].index[0]),
            "top_factor_probability": float(result["latest_probabilities"].iloc[0]),
            "credit_leadership_score": int(credit_score),
            "regime_stability_score": int(stability_score),
            "cv_accuracy": float(result["cv_accuracy"]),
        },
        "factor_probabilities": _series_to_records(result["latest_probabilities"], "probability"),
        "credit_drivers": credit_drivers,
        "risks": risks,
        "dynamic_risks": dynamic_risks,
        "historical_analogs": analogs,
        "feature_importances": _series_to_records(result["feature_importances"].head(25), "importance"),
        "backtest_summary": backtest_summary.to_dict(orient="records"),
        "backtest_metrics": backtest_metrics,
        "validation": validation,
        "data_status": {
            "sources_total": len(source_statuses),
            "sources_successful": len(successful_sources),
            "sources_failed": len(failed_sources),
            "sources": source_statuses,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "latest_dir": str(latest_dir),
            "daily_brief": str(run_dir / "daily_brief.md"),
            "json": str(run_dir / "run_result.json"),
        },
    }

    _write_json(run_dir / "run_result.json", payload)
    _copy_latest(
        run_dir,
        latest_dir,
        root,
        [
            "daily_brief.md",
            "factor_probabilities.csv",
            "regime_scores.csv",
            "backtest_summary.csv",
            "feature_importances.csv",
            "run_result.json",
        ],
    )

    _append_run_history(
        root / "run_history.csv",
        {
            "run_id": run_id,
            "generated_at": payload["generated_at"],
            "regime": regime,
            "top_factor": payload["regime"]["top_factor"],
            "top_factor_probability": payload["regime"]["top_factor_probability"],
            "credit_leadership_score": credit_score,
            "regime_stability_score": stability_score,
            "cv_accuracy": result["cv_accuracy"],
            "sources_failed": len(failed_sources),
        },
    )

    print("Done.")
    print(f"Run ID: {run_id}")
    print(f"Open: {latest_dir / 'daily_brief.md'}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run the Factor Regime Agent.")
    parser.add_argument("--start", default="2003-01-01", help="Start date, e.g. 2003-01-01")
    parser.add_argument("--end", default=None, help="End date, optional, e.g. 2026-06-30")
    parser.add_argument("--horizon", type=int, default=3, help="Forward return horizon in months")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--refresh-data", action="store_true", help="Force fresh FRED/Yahoo downloads instead of using cache")
    parser.add_argument("--run-id", default=None, help="Optional run identifier for reproducible artifact paths")
    args = parser.parse_args()
    run(args.start, args.end, args.horizon, args.output, refresh_data=args.refresh_data, run_id=args.run_id)
