from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_agent.analog import find_historical_analogs
from factor_agent.backtest import factor_backtest_metrics, validate_factor_model
from factor_agent.features import build_factor_excess_returns, forward_factor_winner
from factor_agent.french import combine_academic_and_tradeable_factors
from factor_agent.quality import evaluate_data_quality
from factor_agent.risk import dynamic_regime_risks, regime_break_risk_monitor
from factor_agent.schema import validate_run_result
from factor_agent.scores import credit_leadership_score, regime_label, regime_stability_score


class EngineContractTests(unittest.TestCase):
    def test_factor_excess_returns_and_winner(self) -> None:
        index = pd.date_range("2024-01-31", periods=8, freq="ME")
        prices = pd.DataFrame(
            {
                "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
                "MTUM": [100, 103, 106, 108, 110, 112, 115, 118],
                "QUAL": [100, 101, 103, 104, 105, 106, 107, 108],
            },
            index=index,
        )
        factor_excess = build_factor_excess_returns(prices)
        forward, winner = forward_factor_winner(factor_excess, horizon_months=3)
        self.assertIn("momentum", factor_excess)
        self.assertEqual(winner.dropna().iloc[0], "momentum")
        self.assertEqual(len(forward), len(factor_excess))

    def test_scores_are_bounded(self) -> None:
        latest = pd.Series(
            {
                "hy_oas": 300,
                "hy_oas_1m_chg": -20,
                "hy_oas_3m_chg": -40,
                "ccc_oas_3m_chg": -80,
                "ccc_minus_hy_3m_chg": -60,
                "vix": 18,
                "ism_mfg": 52,
                "cpi_3m_ann": 2.5,
            }
        )
        credit, drivers = credit_leadership_score(latest)
        stability, risks = regime_stability_score(latest)
        self.assertGreaterEqual(credit, 0)
        self.assertLessEqual(credit, 100)
        self.assertGreaterEqual(stability, 0)
        self.assertLessEqual(stability, 100)
        self.assertTrue(drivers)
        self.assertIsInstance(risks, list)

    def test_regime_label_is_stable(self) -> None:
        probs = pd.Series({"value": 0.55, "quality": 0.25, "momentum": 0.20})
        self.assertEqual(regime_label(probs, credit_score=80, stability=80), "Risk-On Expansion")
        self.assertEqual(regime_label(probs, credit_score=50, stability=30), "Transition / Risk-Off Watch")

    def test_analog_and_dynamic_risk_contracts(self) -> None:
        index = pd.date_range("2020-01-31", periods=50, freq="ME")
        features = pd.DataFrame(
            {
                "hy_oas_3m_chg": range(50),
                "vix_1m_chg": [x % 10 for x in range(50)],
                "ism_3m_chg": [5 - (x % 8) for x in range(50)],
                "cpi_3m_ann": [2 + (x % 5) * 0.1 for x in range(50)],
            },
            index=index,
        )
        forward = pd.DataFrame(
            {
                "value": [0.01] * 50,
                "quality": [0.005] * 50,
                "momentum": [-0.002] * 50,
            },
            index=index,
        )
        analogs = find_historical_analogs(features, forward, n=3)
        risks = dynamic_regime_risks(features)
        break_winner = pd.Series((["value", "value", "quality", "quality"] * 13)[:50], index=index)
        break_risks = regime_break_risk_monitor(features, break_winner, "test")
        self.assertIn("analogs", analogs)
        self.assertIn("risks", risks)
        self.assertIn("top_regime_change_risks", break_risks)

    def test_backtest_metrics_contract(self) -> None:
        index = pd.date_range("2023-01-31", periods=6, freq="ME")
        forward = pd.DataFrame({"value": [0.1, -0.1, 0.2, 0.0, 0.1, -0.05]}, index=index)
        winner = pd.Series(["value"] * 6, index=index, name="winner")
        metrics = factor_backtest_metrics(forward, winner)
        self.assertEqual(metrics["observations"], 6)
        self.assertEqual(metrics["factor_metrics"][0]["factor"], "value")

    def test_validation_contract(self) -> None:
        index = pd.date_range("2018-01-31", periods=84, freq="ME")
        features = pd.DataFrame(
            {
                "hy_oas": [300 + (i % 12) for i in range(84)],
                "vix": [15 + (i % 9) for i in range(84)],
                "cpi_3m_ann": [2 + (i % 4) * 0.2 for i in range(84)],
                "ism_mfg": [48 + (i % 8) for i in range(84)],
            },
            index=index,
        )
        factor_excess = pd.DataFrame(
            {
                "value": [0.01 if i % 3 == 0 else -0.002 for i in range(84)],
                "quality": [0.008 if i % 3 == 1 else 0.001 for i in range(84)],
                "momentum": [0.009 if i % 3 == 2 else -0.001 for i in range(84)],
            },
            index=index,
        )
        validation = validate_factor_model(features, factor_excess, horizons=(1, 3), min_train_months=36)
        self.assertEqual(validation["version"], "0.6")
        self.assertIn("validation_window", validation)
        self.assertIn("1", validation["by_horizon"])
        self.assertIn("confusion_matrix", validation["by_horizon"]["1"])
        self.assertIn("baselines", validation["by_horizon"]["1"])

    def test_factor_history_provenance(self) -> None:
        academic_index = pd.date_range("2000-01-31", periods=3, freq="ME")
        tradeable_index = pd.date_range("2000-03-31", periods=2, freq="ME")
        academic = pd.DataFrame({"value": [0.01, 0.02, 0.03]}, index=academic_index)
        tradeable = pd.DataFrame({"value": [0.04, 0.05]}, index=tradeable_index)
        combined, provenance = combine_academic_and_tradeable_factors(academic, tradeable)
        self.assertEqual(combined.loc[pd.Timestamp("2000-01-31"), "value"], 0.01)
        self.assertEqual(combined.loc[pd.Timestamp("2000-03-31"), "value"], 0.04)
        self.assertEqual(provenance["value"]["pre_tradeable_source"], "kenneth_french")

    def test_data_quality_gate(self) -> None:
        statuses = [
            {"name": "hy_oas", "ticker": "BAMLH0A0HYM2", "status": "failed", "latest_observation": None},
            {"name": "SPY", "ticker": "SPY", "status": "live", "latest_observation": "2026-06-12"},
        ]
        quality = evaluate_data_quality(statuses, as_of="2026-06-13")
        self.assertTrue(quality["data_impaired"])
        self.assertLess(quality["confidence_multiplier"], 1.0)

    def test_run_result_schema_validation(self) -> None:
        payload = {
            "schema_version": "0.6",
            "run_id": "test",
            "generated_at": "2026-06-13T12:00:00",
            "parameters": {},
            "regime": {
                "label": "Mixed",
                "top_factor": "value",
                "top_factor_probability": 0.4,
                "adjusted_confidence": 0.3,
            },
            "factor_probabilities": [],
            "data_status": {},
            "data_quality": {"data_impaired": False},
            "factor_history": {},
            "validation": {},
            "artifacts": {},
        }
        validate_run_result(payload)


if __name__ == "__main__":
    unittest.main()
