from __future__ import annotations

import numpy as np
import pandas as pd
from .config import FACTOR_NAMES


def build_monthly_features(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.resample("ME").last().ffill()

    f = pd.DataFrame(index=m.index)
    # Levels
    for col in m.columns:
        f[col] = m[col]

    # Rates and credit changes
    if {"dgs10", "dgs2"}.issubset(m.columns):
        f["curve_2s10s"] = m["dgs10"] - m["dgs2"]
        f["curve_2s10s_3m_chg"] = f["curve_2s10s"].diff(3)
    if "dgs10" in m:
        f["dgs10_1m_chg"] = m["dgs10"].diff(1)
        f["dgs10_3m_chg"] = m["dgs10"].diff(3)
    if "tips10" in m:
        f["real_yield_3m_chg"] = m["tips10"].diff(3)

    for col in ["hy_oas", "ig_oas", "ccc_oas", "vix"]:
        if col in m:
            f[f"{col}_1m_chg"] = m[col].diff(1)
            f[f"{col}_3m_chg"] = m[col].diff(3)
            f[f"{col}_6m_chg"] = m[col].diff(6)

    if {"ccc_oas", "hy_oas"}.issubset(m.columns):
        f["ccc_minus_hy"] = m["ccc_oas"] - m["hy_oas"]
        f["ccc_minus_hy_3m_chg"] = f["ccc_minus_hy"].diff(3)

    # Macro trend proxies
    if "ism_mfg" in m:
        f["ism_3m_chg"] = m["ism_mfg"].diff(3)
        f["ism_above_50"] = (m["ism_mfg"] > 50).astype(float)
    if "claims" in m:
        f["claims_1m_pct_chg"] = m["claims"].pct_change(1)
        f["claims_3m_pct_chg"] = m["claims"].pct_change(3)
    if "payrolls" in m:
        f["payrolls_3m_pct_chg"] = m["payrolls"].pct_change(3)
    if "cpi" in m:
        f["cpi_3m_ann"] = ((m["cpi"] / m["cpi"].shift(3)) ** 4 - 1) * 100
        f["cpi_6m_ann"] = ((m["cpi"] / m["cpi"].shift(6)) ** 2 - 1) * 100

    return f.replace([np.inf, -np.inf], np.nan)


def build_factor_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = prices.resample("ME").last().ffill()
    returns = monthly_prices.pct_change(fill_method=None)
    if "SPY" not in returns:
        raise ValueError("SPY is required as the benchmark.")

    out = pd.DataFrame(index=returns.index)
    for factor, ticker in FACTOR_NAMES.items():
        if ticker in returns:
            out[factor] = returns[ticker] - returns["SPY"]
    return out.dropna(how="all")


def forward_factor_winner(factor_excess: pd.DataFrame, horizon_months: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    forward = factor_excess.rolling(horizon_months).sum().shift(-horizon_months)
    winner = forward.dropna(how="all").idxmax(axis=1)
    return forward, winner.rename("winner")
