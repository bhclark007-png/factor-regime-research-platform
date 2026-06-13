from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


FRENCH_5_FACTOR_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FRENCH_MOMENTUM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


@dataclass
class FactorHistory:
    returns: pd.DataFrame
    metadata: dict
    statuses: list[dict]


def _download_zip(url: str, cache_path: Path, refresh: bool = False) -> tuple[bytes, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes(), "cache"
    response = requests.get(url, timeout=60, headers={"User-Agent": "factor-regime-agent/0.1"})
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.content, "live"


def _parse_french_monthly_zip(zip_bytes: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(zip_bytes)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin1")

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(","):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find Kenneth French monthly table header.")

    table_lines = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if stripped.lower().startswith("annual"):
            break
        table_lines.append(line)

    frame = pd.read_csv(StringIO("\n".join(table_lines)))
    first_col = frame.columns[0]
    frame = frame.rename(columns={first_col: "date"})
    frame = frame[frame["date"].astype(str).str.match(r"^\d{6}$", na=False)].copy()
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = frame.set_index("date").sort_index()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce") / 100.0
    return frame.dropna(how="all")


def get_kenneth_french_factors(
    start: str,
    end: str | None = None,
    cache_dir: str | Path = ".cache/data",
    refresh: bool = False,
) -> FactorHistory:
    """Load academic monthly factor returns from Kenneth French's data library.

    Returns are monthly decimal excess returns. The series are academic factor
    portfolios, not directly tradeable ETF returns.
    """
    cache_root = Path(cache_dir) / "kenneth_french"
    statuses = []

    five_bytes, five_status = _download_zip(FRENCH_5_FACTOR_URL, cache_root / "ff5.zip", refresh=refresh)
    five = _parse_french_monthly_zip(five_bytes)
    statuses.append(
        {
            "source": "kenneth_french",
            "name": "ff5",
            "ticker": "F-F_Research_Data_5_Factors_2x3",
            "status": five_status,
            "rows": int(len(five)),
            "latest_observation": five.index.max().strftime("%Y-%m-%d") if not five.empty else None,
        }
    )

    momentum_bytes, momentum_status = _download_zip(FRENCH_MOMENTUM_URL, cache_root / "momentum.zip", refresh=refresh)
    momentum = _parse_french_monthly_zip(momentum_bytes)
    statuses.append(
        {
            "source": "kenneth_french",
            "name": "momentum",
            "ticker": "F-F_Momentum_Factor",
            "status": momentum_status,
            "rows": int(len(momentum)),
            "latest_observation": momentum.index.max().strftime("%Y-%m-%d") if not momentum.empty else None,
        }
    )

    out = pd.DataFrame(index=five.index.union(momentum.index).sort_values())
    if "SMB" in five:
        out["small_cap"] = five["SMB"]
    if "HML" in five:
        out["value"] = five["HML"]
    if {"RMW", "CMA"}.issubset(five.columns):
        out["quality"] = (five["RMW"] - five["CMA"]) / 2.0
    elif "RMW" in five:
        out["quality"] = five["RMW"]
    mom_col = "Mom" if "Mom" in momentum.columns else momentum.columns[0]
    out["momentum"] = momentum[mom_col]
    out = out.sort_index()
    out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]

    metadata = {
        "source_type": "academic_factor_portfolio",
        "tradeable": False,
        "description": "Kenneth French academic monthly factor portfolios used to extend research history before ETF proxy inception.",
        "available_factors": list(out.columns),
        "missing_tradeable_factor_proxy": ["low_vol"],
        "mapping": {
            "small_cap": "SMB",
            "value": "HML",
            "quality": "RMW minus CMA composite",
            "momentum": "Momentum factor",
        },
    }
    return FactorHistory(out, metadata, statuses)


def combine_academic_and_tradeable_factors(
    academic: pd.DataFrame,
    tradeable: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Use academic history before ETF proxy availability and ETF proxies after."""
    combined = academic.copy()
    provenance = {}
    all_columns = sorted(set(academic.columns) | set(tradeable.columns))
    all_index = academic.index.union(tradeable.index).sort_values()
    combined = combined.reindex(all_index)

    for factor in all_columns:
        academic_series = academic[factor] if factor in academic else pd.Series(index=all_index, dtype=float)
        tradeable_series = tradeable[factor] if factor in tradeable else pd.Series(index=all_index, dtype=float)
        combined[factor] = academic_series.reindex(all_index).combine_first(pd.Series(index=all_index, dtype=float))
        combined[factor] = tradeable_series.reindex(all_index).combine_first(combined[factor])
        first_tradeable = tradeable_series.dropna().index.min() if not tradeable_series.dropna().empty else None
        provenance[factor] = {
            "pre_tradeable_source": "kenneth_french" if factor in academic else None,
            "tradeable_proxy_source": "etf" if factor in tradeable else None,
            "first_tradeable_observation": first_tradeable.strftime("%Y-%m-%d") if first_tradeable is not None else None,
            "is_tradeable_latest": bool(factor in tradeable and tradeable_series.dropna().index.max() == combined[factor].dropna().index.max()),
        }

    return combined.dropna(how="all"), provenance
