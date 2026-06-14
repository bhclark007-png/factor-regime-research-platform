from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_agent.analog import find_historical_analogs
from factor_agent.backtest import factor_backtest_metrics, validate_factor_model
from factor_agent.agent import _select_factor_returns, run
from factor_agent.brief import make_daily_brief
from factor_agent.data import SourceStatus
from factor_agent.features import build_factor_excess_returns, forward_factor_winner
from factor_agent.french import FactorHistory, combine_academic_and_tradeable_factors
from factor_agent.quality import evaluate_data_quality
from factor_agent.risk import (
    dynamic_regime_risks,
    identify_regime_breaks,
    regime_break_risk_monitor,
)
from factor_agent.schema import validate_run_result
from factor_agent.scores import (
    credit_leadership_score,
    regime_label,
    regime_stability_score,
)


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
        self.assertEqual(
            regime_label(probs, credit_score=80, stability=80), "Risk-On Expansion"
        )
        self.assertEqual(
            regime_label(probs, credit_score=50, stability=30),
            "Transition / Risk-Off Watch",
        )

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
        break_winner = pd.Series(
            (["value", "value", "quality", "quality"] * 13)[:50], index=index
        )
        break_risks = regime_break_risk_monitor(features, break_winner, "test")
        self.assertIn("analogs", analogs)
        self.assertIn("risks", risks)
        self.assertIn("top_regime_change_risks", break_risks)

    def test_backtest_metrics_contract(self) -> None:
        index = pd.date_range("2023-01-31", periods=6, freq="ME")
        forward = pd.DataFrame(
            {"value": [0.1, -0.1, 0.2, 0.0, 0.1, -0.05]}, index=index
        )
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
        validation = validate_factor_model(
            features, factor_excess, horizons=(1, 3), min_train_months=36
        )
        self.assertEqual(validation["version"], "0.6.1")
        self.assertIn("validation_window", validation)
        self.assertIn("1", validation["by_horizon"])
        self.assertIn("confusion_matrix", validation["by_horizon"]["1"])
        self.assertIn("baselines", validation["by_horizon"]["1"])
        self.assertIn("model_value_add", validation["by_horizon"]["1"])
        compared = {
            row["baseline"] for row in validation["by_horizon"]["1"]["model_value_add"]
        }
        self.assertIn("previous_winner", compared)
        self.assertIn("equal_weight_factors", compared)
        self.assertIn("spy", compared)

    def test_factor_history_provenance(self) -> None:
        academic_index = pd.date_range("2000-01-31", periods=3, freq="ME")
        tradeable_index = pd.date_range("2000-03-31", periods=2, freq="ME")
        academic = pd.DataFrame({"value": [0.01, 0.02, 0.03]}, index=academic_index)
        tradeable = pd.DataFrame({"value": [0.04, 0.05]}, index=tradeable_index)
        combined, provenance = combine_academic_and_tradeable_factors(
            academic, tradeable
        )
        self.assertEqual(combined.loc[pd.Timestamp("2000-01-31"), "value"], 0.01)
        self.assertEqual(combined.loc[pd.Timestamp("2000-03-31"), "value"], 0.04)
        self.assertEqual(provenance["value"]["pre_tradeable_source"], "kenneth_french")

    def test_data_quality_gate(self) -> None:
        statuses = [
            {
                "name": "hy_oas",
                "ticker": "BAMLH0A0HYM2",
                "status": "failed",
                "latest_observation": None,
            },
            {
                "name": "SPY",
                "ticker": "SPY",
                "status": "live",
                "latest_observation": "2026-06-12",
            },
        ]
        quality = evaluate_data_quality(statuses, as_of="2026-06-13")
        self.assertTrue(quality["data_impaired"])
        self.assertLess(quality["confidence_multiplier"], 1.0)

    def test_run_result_schema_validation(self) -> None:
        payload = {
            "schema_version": "0.6.1",
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
            "factor_history": {"selected_mode": "tradeable"},
            "validation": {"version": "0.6.1"},
            "dynamic_risks": {"method": "stress_percentile_monitoring"},
            "regime_break_risks": {"method": "historical_regime_break_monitoring"},
            "artifacts": {},
        }
        validate_run_result(payload)

    def test_daily_brief_generation_accepts_data_quality(self) -> None:
        probabilities = pd.Series({"value": 0.5, "quality": 0.3, "momentum": 0.2})
        brief = make_daily_brief(
            probabilities=probabilities,
            cv_accuracy=0.4,
            credit_score=60,
            credit_drivers=["HY spreads are contained."],
            stability_score=70,
            risks=[],
            dynamic_risks={"transition_probability": None, "risks": []},
            regime="Risk-On Expansion",
            analogs={"analogs": []},
            data_quality={"data_impaired": False},
            feature_importances=pd.Series({"hy_oas": 0.2}),
        )
        self.assertIn("Daily Factor Regime Brief", brief)

    def test_factor_source_mode_selection(self) -> None:
        index = pd.date_range("2020-01-31", periods=4, freq="ME")
        academic = pd.DataFrame({"value": [0.01, 0.02, 0.03, 0.04]}, index=index)
        tradeable = pd.DataFrame({"value": [0.10, 0.20]}, index=index[-2:])

        academic_selected, academic_meta = _select_factor_returns(
            "academic", academic, tradeable
        )
        tradeable_selected, tradeable_meta = _select_factor_returns(
            "tradeable", academic, tradeable
        )
        combined_selected, combined_meta = _select_factor_returns(
            "combined", academic, tradeable
        )

        self.assertEqual(academic_meta["selected_mode"], "academic")
        self.assertEqual(tradeable_meta["selected_mode"], "tradeable")
        self.assertEqual(combined_meta["selected_mode"], "combined")
        self.assertEqual(len(academic_selected), 4)
        self.assertEqual(len(tradeable_selected), 2)
        self.assertEqual(combined_selected.loc[index[-1], "value"], 0.20)

    def test_risk_outputs_are_self_describing(self) -> None:
        index = pd.date_range("2015-01-31", periods=96, freq="ME")
        features = pd.DataFrame(
            {
                "hy_oas_3m_chg": [i % 20 for i in range(96)],
                "vix_1m_chg": [i % 15 for i in range(96)],
                "ccc_minus_hy_3m_chg": [i % 10 for i in range(96)],
                "ism_3m_chg": [10 - (i % 12) for i in range(96)],
            },
            index=index,
        )
        winner = pd.Series((["value"] * 6 + ["quality"] * 6) * 8, index=index)
        stress = dynamic_regime_risks(features)
        breaks = regime_break_risk_monitor(features, winner, "test")

        self.assertEqual(stress["method"], "stress_percentile_monitoring")
        self.assertEqual(breaks["method"], "historical_regime_break_monitoring")
        self.assertIn("defined_break_count", breaks)
        for risk in breaks["top_regime_change_risks"]:
            self.assertIn("historical_frequency_before_transitions", risk)
            self.assertIn("severity_percentile", risk)

    def test_regime_break_ranking_uses_transition_windows(self) -> None:
        index = pd.date_range("2010-01-31", periods=120, freq="ME")
        winners = pd.Series(
            (["value"] * 12 + ["quality"] * 12) * 5,
            index=index,
            name="winner",
        )
        hy_stress = pd.Series(0.0, index=index)
        vix_stress = pd.Series(0.0, index=index)

        for block_start in range(0, len(index), 12):
            block = index[block_start : block_start + 12]
            if len(block) < 12:
                continue
            hy_stress.loc[block[9:12]] = 100.0
            if block_start > 0:
                hy_stress.loc[block[0]] = 100.0
            vix_stress.loc[block[2:6]] = 100.0

        features = pd.DataFrame(
            {
                "hy_oas_3m_chg": hy_stress,
                "vix_1m_chg": vix_stress,
            },
            index=index,
        )
        breaks = identify_regime_breaks(winners, features)
        risks = regime_break_risk_monitor(features, winners, "test", top_n=2)

        self.assertGreater(len(breaks), 0)
        self.assertEqual(
            risks["top_regime_change_risks"][0]["indicator"], "hy_oas_3m_chg"
        )
        hy_frequency = risks["top_regime_change_risks"][0][
            "historical_frequency_before_transitions"
        ]
        other_frequency = risks["top_regime_change_risks"][1][
            "historical_frequency_before_transitions"
        ]
        self.assertGreater(hy_frequency, other_frequency)

    def test_run_agent_completes_and_emits_valid_schema(self) -> None:
        index = pd.date_range("2018-01-31", periods=72, freq="ME")
        features = pd.DataFrame(
            {
                "hy_oas": [300 + i for i in range(72)],
                "hy_oas_3m_chg": [i % 10 for i in range(72)],
                "ccc_minus_hy_3m_chg": [i % 7 for i in range(72)],
                "vix": [15 + (i % 5) for i in range(72)],
                "vix_1m_chg": [i % 4 for i in range(72)],
                "ism_mfg": [50 + (i % 3) for i in range(72)],
                "ism_3m_chg": [2 - (i % 4) for i in range(72)],
                "cpi_3m_ann": [2.0 + (i % 4) * 0.1 for i in range(72)],
                "curve_2s10s_3m_chg": [0.1 * (i % 5) for i in range(72)],
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "SPY": [100 + i for i in range(72)],
                "MTUM": [100 + i * 1.2 for i in range(72)],
                "QUAL": [100 + i * 1.1 for i in range(72)],
                "VLUE": [100 + i * 1.3 for i in range(72)],
                "USMV": [100 + i * 0.9 for i in range(72)],
                "IWM": [100 + i * 1.05 for i in range(72)],
            },
            index=index,
        )
        academic = pd.DataFrame(
            {
                "value": [0.01 if i % 2 else -0.002 for i in range(72)],
                "quality": [0.006 for _ in range(72)],
                "momentum": [0.004 for _ in range(72)],
                "small_cap": [0.003 for _ in range(72)],
            },
            index=index,
        )
        statuses = [
            SourceStatus(
                "test", "SPY", "SPY", "live", rows=72, latest_observation="2026-06-01"
            )
        ]
        train_result = {
            "X": features,
            "y": pd.Series(
                ["value", "quality", "momentum"] * 24, index=index, name="winner"
            ),
            "cv_accuracy": 0.4,
            "latest_probabilities": pd.Series(
                {"value": 0.45, "quality": 0.35, "momentum": 0.20}
            ),
            "feature_importances": pd.Series({"hy_oas": 0.3, "vix": 0.2}),
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("factor_agent.agent.get_fred", return_value=(features, statuses)),
                patch(
                    "factor_agent.agent.get_etf_prices", return_value=(prices, statuses)
                ),
                patch(
                    "factor_agent.agent.get_kenneth_french_factors",
                    return_value=FactorHistory(
                        academic,
                        {
                            "source_type": "academic_factor_portfolio",
                            "available_factors": list(academic.columns),
                        },
                        [
                            {
                                "source": "kenneth_french",
                                "name": "ff5",
                                "ticker": "ff5",
                                "status": "cache",
                                "rows": 72,
                            }
                        ],
                    ),
                ),
                patch(
                    "factor_agent.agent.train_factor_model", return_value=train_result
                ),
                patch(
                    "factor_agent.agent.validate_factor_model",
                    return_value={"version": "0.6.1", "by_horizon": {}, "summary": []},
                ),
            ):
                payload = run(
                    "2018-01-01",
                    None,
                    3,
                    tmp,
                    run_id="unit_run",
                    factor_source="tradeable",
                )

            validate_run_result(payload)
            self.assertEqual(payload["factor_history"]["selected_mode"], "tradeable")
            self.assertTrue((Path(tmp) / "latest" / "run_result.json").exists())
            self.assertTrue((Path(tmp) / "latest" / "daily_brief.md").exists())

    def test_academic_mode_wires_kenneth_french_into_validation(self) -> None:
        index = pd.date_range("2018-01-31", periods=72, freq="ME")
        features = pd.DataFrame(
            {
                "hy_oas": [300 + i for i in range(72)],
                "hy_oas_3m_chg": [i % 10 for i in range(72)],
                "vix": [15 + (i % 5) for i in range(72)],
                "vix_1m_chg": [i % 4 for i in range(72)],
                "ism_mfg": [50 + (i % 3) for i in range(72)],
                "cpi_3m_ann": [2.0 + (i % 4) * 0.1 for i in range(72)],
            },
            index=index,
        )
        prices = pd.DataFrame(
            {
                "SPY": [100 + i for i in range(72)],
                "MTUM": [100 + i * 1.2 for i in range(72)],
                "QUAL": [100 + i * 1.1 for i in range(72)],
                "VLUE": [100 + i * 1.3 for i in range(72)],
                "USMV": [100 + i * 0.9 for i in range(72)],
                "IWM": [100 + i * 1.05 for i in range(72)],
            },
            index=index,
        )
        academic = pd.DataFrame(
            {
                "value": [0.01 if i % 2 else -0.002 for i in range(72)],
                "quality": [0.006 for _ in range(72)],
                "momentum": [0.004 for _ in range(72)],
                "small_cap": [0.003 for _ in range(72)],
            },
            index=index,
        )
        statuses = [
            SourceStatus(
                "test", "SPY", "SPY", "live", rows=72, latest_observation="2026-06-01"
            )
        ]
        train_result = {
            "X": features,
            "y": pd.Series(
                ["value", "quality", "momentum"] * 24, index=index, name="winner"
            ),
            "cv_accuracy": 0.4,
            "latest_probabilities": pd.Series(
                {"value": 0.45, "quality": 0.35, "momentum": 0.20}
            ),
            "feature_importances": pd.Series({"hy_oas": 0.3, "vix": 0.2}),
        }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("factor_agent.agent.get_fred", return_value=(features, statuses)),
                patch(
                    "factor_agent.agent.get_etf_prices", return_value=(prices, statuses)
                ),
                patch(
                    "factor_agent.agent.get_kenneth_french_factors",
                    return_value=FactorHistory(
                        academic,
                        {
                            "source_type": "academic_factor_portfolio",
                            "available_factors": list(academic.columns),
                        },
                        [
                            {
                                "source": "kenneth_french",
                                "name": "ff5",
                                "ticker": "ff5",
                                "status": "cache",
                                "rows": 72,
                            }
                        ],
                    ),
                ),
                patch(
                    "factor_agent.agent.train_factor_model", return_value=train_result
                ),
                patch(
                    "factor_agent.agent.validate_factor_model",
                    return_value={"version": "0.6.1", "by_horizon": {}, "summary": []},
                ) as validation_mock,
            ):
                payload = run(
                    "2018-01-01",
                    None,
                    3,
                    tmp,
                    run_id="academic_unit_run",
                    factor_source="academic",
                )

        validation_factor_excess = validation_mock.call_args.args[1]
        self.assertEqual(payload["factor_history"]["selected_mode"], "academic")
        self.assertEqual(
            payload["factor_history"]["model_series"], "kenneth_french_academic"
        )
        self.assertEqual(set(validation_factor_excess.columns), set(academic.columns))
        self.assertNotIn("low_vol", validation_factor_excess.columns)


if __name__ == "__main__":
    unittest.main()
