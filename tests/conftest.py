from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import pytest

from factor_agent.data import SourceStatus
from factor_agent.french import FactorHistory

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read_fixture_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, index_col="date", parse_dates=True)


@pytest.fixture
def sample_macro() -> pd.DataFrame:
    return _read_fixture_csv("sample_macro.csv")


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    return _read_fixture_csv("sample_prices.csv")


@pytest.fixture
def sample_french_returns() -> pd.DataFrame:
    return _read_fixture_csv("sample_french_factors.csv")


@pytest.fixture
def sample_french_history(sample_french_returns: pd.DataFrame) -> FactorHistory:
    return FactorHistory(
        returns=sample_french_returns,
        metadata={
            "source_type": "academic_factor_portfolio",
            "available_factors": list(sample_french_returns.columns),
            "description": "Fixture Kenneth French-style academic factors.",
        },
        statuses=[
            {
                "source": "kenneth_french",
                "name": "fixture",
                "ticker": "fixture",
                "status": "cache",
                "rows": len(sample_french_returns),
                "latest_observation": sample_french_returns.index.max().strftime(
                    "%Y-%m-%d"
                ),
            }
        ],
    )


@pytest.fixture
def source_statuses(sample_macro: pd.DataFrame) -> list[SourceStatus]:
    latest = sample_macro.index.max().strftime("%Y-%m-%d")
    return [
        SourceStatus(
            "fixture",
            "SPY",
            "SPY",
            "cache",
            rows=len(sample_macro),
            latest_observation=latest,
        ),
        SourceStatus(
            "fixture",
            "hy_oas",
            "hy_oas",
            "cache",
            rows=len(sample_macro),
            latest_observation=latest,
        ),
    ]


@pytest.fixture
def fake_train_result(sample_macro: pd.DataFrame) -> dict:
    index = sample_macro.index
    return {
        "X": pd.DataFrame(
            {
                "hy_oas": sample_macro["hy_oas"],
                "hy_oas_3m_chg": sample_macro["hy_oas"].diff(3).fillna(0),
                "vix": sample_macro["vix"],
                "vix_1m_chg": sample_macro["vix"].diff(1).fillna(0),
                "ism_3m_chg": sample_macro["ism_mfg"].diff(3).fillna(0),
                "cpi_3m_ann": [2.0 + (i % 4) * 0.1 for i in range(len(index))],
            },
            index=index,
        ),
        "y": pd.Series(
            ["value", "quality", "momentum", "small_cap"] * 3,
            index=index,
            name="winner",
        ),
        "cv_accuracy": 0.35,
        "latest_probabilities": pd.Series(
            {"value": 0.42, "quality": 0.25, "momentum": 0.20, "small_cap": 0.13}
        ),
        "feature_importances": pd.Series(
            {"hy_oas": 0.3, "vix": 0.2, "ism_3m_chg": 0.1}
        ),
    }


@pytest.fixture
def fixture_run_result_path() -> Path:
    return FIXTURES / "sample_run_result.json"
