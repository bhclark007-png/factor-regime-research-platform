from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
import re
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader.fred import FredReader
from yfinance import cache as yf_cache

TREASURY_XML_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
VIX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
TRADING_ECONOMICS_HISTORICAL_URL = "https://api.tradingeconomics.com/historical/country/united states/indicator/ism manufacturing pmi"
TRADING_ECONOMICS_CCC_INDICATORS = [
    "bofa merrill lynch us high yield ccc or below option adjusted spread",
    "ice bofa ccc lower us high yield index option-adjusted spread",
    "ice bofa ccc & lower us high yield index option-adjusted spread",
]

FRED_ALTERNATES = {
    "dgs2": {
        "source": "treasury",
        "data": "daily_treasury_yield_curve",
        "field": "BC_2YEAR",
    },
    "dgs10": {
        "source": "treasury",
        "data": "daily_treasury_yield_curve",
        "field": "BC_10YEAR",
    },
    "tips10": {
        "source": "treasury",
        "data": "daily_treasury_real_yield_curve",
        "field": "TC_10YEAR",
    },
    "vix": {"source": "cboe", "field": "CLOSE"},
    "cpi": {"source": "bls", "series_id": "CUSR0000SA0"},
    "payrolls": {"source": "bls", "series_id": "CES0000000001"},
    "ism_mfg": {"source": "manufacturing_growth_proxy", "series_id": "IPMAN"},
    "ccc_oas": {"source": "ccc_oas_seed", "ticker": "BAMLH0A3HYC"},
}


@dataclass
class SourceStatus:
    source: str
    name: str
    ticker: str
    status: str
    rows: int = 0
    cache_path: str | None = None
    error: str | None = None
    fetched_at: str | None = None
    latest_observation: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_path(
    cache_dir: Path, source: str, ticker: str, start: str, end: str | None
) -> Path:
    end_key = end or "latest"
    return (
        cache_dir
        / source
        / f"{_safe_name(ticker)}_{_safe_name(start)}_{_safe_name(end_key)}.csv"
    )


def _read_cached_series(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if column not in df:
        raise RuntimeError(f"Cache file is missing expected column {column}")
    return df[[column]]


def _write_cached_series(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _latest_observation(frame: pd.DataFrame | pd.Series) -> str | None:
    if frame.empty:
        return None
    index = (
        frame.dropna(how="all").index
        if isinstance(frame, pd.DataFrame)
        else frame.dropna().index
    )
    if len(index) == 0:
        return None
    return pd.Timestamp(index.max()).strftime("%Y-%m-%d")


def _years_between(start: str, end: str | None) -> range:
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year if end else datetime.now().year
    return range(start_year, end_year + 1)


def _filter_date_range(
    frame: pd.DataFrame, start: str, end: str | None
) -> pd.DataFrame:
    out = frame.loc[frame.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _fetch_treasury_series(
    data_key: str, field: str, start: str, end: str | None
) -> pd.Series:
    rows = []
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    }
    for year in _years_between(start, end):
        resp = requests.get(
            TREASURY_XML_URL,
            params={"data": data_key, "field_tdr_date_value": str(year)},
            timeout=30,
            headers={"User-Agent": "factor-regime-agent/0.1"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for props in root.findall("a:entry/a:content/m:properties", ns):
            values = {child.tag.split("}")[-1]: child.text for child in props}
            if values.get("NEW_DATE") and values.get(field):
                rows.append(
                    (pd.Timestamp(values["NEW_DATE"]).normalize(), float(values[field]))
                )

    if not rows:
        raise RuntimeError(f"Treasury returned no rows for {data_key}/{field}")
    series = pd.Series(dict(rows)).sort_index()
    return _filter_date_range(series.to_frame("value"), start, end)["value"]


def _fetch_cboe_vix(start: str, end: str | None) -> pd.Series:
    frame = pd.read_csv(VIX_HISTORY_URL)
    frame.columns = [str(c).upper().strip() for c in frame.columns]
    date_col = "DATE" if "DATE" in frame.columns else frame.columns[0]
    value_col = "CLOSE" if "CLOSE" in frame.columns else "VIX CLOSE"
    if value_col not in frame.columns:
        raise RuntimeError("Cboe VIX CSV did not include a close column")
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    return _filter_date_range(
        frame[[value_col]].rename(columns={value_col: "value"}), start, end
    )["value"]


def _fetch_bls_series(series_id: str, start: str, end: str | None) -> pd.Series:
    rows = []
    for first_year in range(
        pd.Timestamp(start).year, (_years_between(start, end).stop), 20
    ):
        last_year = min(first_year + 19, (_years_between(start, end).stop - 1))
        resp = requests.post(
            BLS_API_URL,
            json={
                "seriesid": [series_id],
                "startyear": str(first_year),
                "endyear": str(last_year),
            },
            timeout=30,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "factor-regime-agent/0.1",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS request failed: {payload.get('message')}")
        series = payload.get("Results", {}).get("series", [])
        if not series:
            continue
        for item in series[0].get("data", []):
            period = item.get("period", "")
            if not period.startswith("M"):
                continue
            date = pd.Timestamp(int(item["year"]), int(period[1:]), 1)
            try:
                value = float(item["value"])
            except (TypeError, ValueError):
                continue
            rows.append((date, value))

    if not rows:
        raise RuntimeError(f"BLS returned no rows for {series_id}")
    series = pd.Series(dict(rows)).sort_index()
    return _filter_date_range(series.to_frame("value"), start, end)["value"]


def _fetch_trading_economics_ism_pmi(start: str, end: str | None) -> pd.Series | None:
    api_key = os.environ.get("TRADING_ECONOMICS_API_KEY")
    if not api_key:
        return None

    resp = requests.get(
        TRADING_ECONOMICS_HISTORICAL_URL,
        params={"c": api_key},
        timeout=30,
        headers={"User-Agent": "factor-regime-agent/0.1"},
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Trading Economics returned no ISM PMI rows")

    rows = []
    for item in payload:
        date_value = item.get("DateTime") or item.get("date") or item.get("Date")
        close_value = item.get("Close") or item.get("close") or item.get("Value")
        if date_value is None or close_value is None:
            continue
        rows.append((pd.Timestamp(date_value).normalize(), float(close_value)))

    if not rows:
        raise RuntimeError(
            "Trading Economics response did not contain DateTime/Close observations"
        )

    series = pd.Series(dict(rows)).sort_index().rename("ism_mfg")
    series = _filter_date_range(series.to_frame("value"), start, end)["value"]
    series.attrs["source"] = "trading_economics"
    return series


def _fetch_trading_economics_indicator(
    indicators: list[str], start: str, end: str | None
) -> pd.Series | None:
    api_key = os.environ.get("TRADING_ECONOMICS_API_KEY")
    if not api_key:
        return None

    last_error = None
    for indicator in indicators:
        try:
            url = f"https://api.tradingeconomics.com/historical/country/united states/indicator/{indicator}"
            resp = requests.get(
                url,
                params={"c": api_key},
                timeout=30,
                headers={"User-Agent": "factor-regime-agent/0.1"},
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                continue

            rows = []
            for item in payload:
                date_value = (
                    item.get("DateTime") or item.get("date") or item.get("Date")
                )
                close_value = (
                    item.get("Close") or item.get("close") or item.get("Value")
                )
                if date_value is None or close_value is None:
                    continue
                rows.append((pd.Timestamp(date_value).normalize(), float(close_value)))

            if rows:
                series = pd.Series(dict(rows)).sort_index()
                series = _filter_date_range(series.to_frame("value"), start, end)[
                    "value"
                ]
                series.attrs["source"] = "trading_economics"
                return series
        except Exception as exc:
            last_error = exc
    if last_error:
        raise RuntimeError(f"Trading Economics CCC fetch failed: {last_error}")
    return None


def _read_local_series_file(
    path: Path, value_name: str, start: str, end: str | None
) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = {c.lower(): c for c in frame.columns}
    date_col = (
        columns.get("date") or columns.get("observation_date") or frame.columns[0]
    )
    value_col = (
        columns.get(value_name.lower())
        or columns.get("value")
        or columns.get("close")
        or frame.columns[-1]
    )
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[value_col]).set_index(date_col).sort_index()
    series = _filter_date_range(
        frame[[value_col]].rename(columns={value_col: "value"}), start, end
    )["value"]
    series.attrs["source"] = "local_seed"
    return series


def _fetch_ccc_oas_series(ticker: str, start: str, end: str | None) -> pd.Series:
    trading_economics = _fetch_trading_economics_indicator(
        TRADING_ECONOMICS_CCC_INDICATORS, start, end
    )
    if trading_economics is not None and not trading_economics.empty:
        trading_economics.attrs["source"] = "trading_economics"
        return trading_economics

    seed_path = Path("data") / "ccc_oas_history.csv"
    seed = _read_local_series_file(seed_path, "ccc_oas", start, end)

    latest = None
    latest_error = None
    try:
        latest_df = _get_fred_series(ticker, start, end, retries=2, timeout=90)
        latest = latest_df.iloc[:, 0].dropna()
    except Exception as exc:
        latest_error = exc

    if seed is not None and not seed.empty:
        if latest is not None and not latest.empty:
            combined = pd.concat([seed, latest]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            combined.rename("ccc_oas").to_csv(seed_path, index_label="DATE")
            combined.attrs["source"] = "local_seed_plus_fred_update"
            return combined
        seed.attrs["source"] = "local_seed"
        return seed

    if latest is not None and not latest.empty:
        latest.attrs["source"] = "fred_recent"
        return _filter_date_range(latest.to_frame("value"), start, end)["value"]

    raise RuntimeError(
        "No CCC OAS seed data available. Add data/ccc_oas_history.csv or set "
        f"TRADING_ECONOMICS_API_KEY. Latest FRED error: {latest_error}"
    )


def _fetch_manufacturing_growth_proxy(
    series_id: str, start: str, end: str | None
) -> pd.Series:
    trading_economics = _fetch_trading_economics_ism_pmi(start, end)
    if trading_economics is not None and not trading_economics.empty:
        return trading_economics

    raw_start = (pd.Timestamp(start) - pd.DateOffset(years=12)).strftime("%Y-%m-%d")
    ip = _get_fred_series(series_id, raw_start, end, retries=2, timeout=30)
    ip_series = ip.iloc[:, 0].sort_index().dropna()

    growth_3m_ann = ip_series.pct_change(3) * 400
    rolling_mean = growth_3m_ann.rolling(120, min_periods=36).mean()
    rolling_std = growth_3m_ann.rolling(120, min_periods=36).std()
    z_score = ((growth_3m_ann - rolling_mean) / rolling_std).clip(-2.5, 2.5)

    proxy = (50 + 5 * z_score).rename("ism_mfg")
    proxy = proxy.ffill().dropna()
    proxy = _filter_date_range(proxy.to_frame("value"), start, end)["value"]

    latest_ism = _scrape_latest_ism_pmi()
    if latest_ism:
        date, value = latest_ism
        if date >= pd.Timestamp(start) and (end is None or date <= pd.Timestamp(end)):
            proxy.loc[date] = value
            proxy = proxy.sort_index()
            proxy.attrs["latest_overlay"] = "ism_public_report"
    proxy.attrs["source"] = "manufacturing_growth_proxy"
    return proxy


def _scrape_latest_ism_pmi() -> tuple[pd.Timestamp, float] | None:
    month_names = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    now = datetime.now()
    candidates = []
    for offset in range(0, 4):
        month_date = pd.Timestamp(now.year, now.month, 1) - pd.DateOffset(months=offset)
        month = month_names[month_date.month - 1]
        candidates.extend(
            [
                f"https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/pmi/{month}/",
                f"https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/{month}/",
            ]
        )

    patterns = [
        r"Manufacturing PMI[^0-9]{0,80}(?:registered|at)\s+([0-9]+(?:\.[0-9]+)?)",
        r"PMI[^0-9]{0,80}(?:registered|at)\s+([0-9]+(?:\.[0-9]+)?)\s*percent",
    ]

    for url in candidates:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            text = resp.text
            if (
                resp.status_code >= 400
                or "captcha_form" in text.lower()
                or "SSO/Login" in resp.url
            ):
                continue
            plain = re.sub(r"\s+", " ", text)
            for pattern in patterns:
                match = re.search(pattern, plain, flags=re.IGNORECASE)
                if match:
                    month = url.rstrip("/").split("/")[-1].lower()
                    month_num = month_names.index(month) + 1
                    date = pd.Timestamp(now.year, month_num, 1)
                    if month_num > now.month:
                        date = pd.Timestamp(now.year - 1, month_num, 1)
                    return date, float(match.group(1))
        except Exception:
            continue
    return None


def _get_alternate_series(
    name: str, start: str, end: str | None
) -> tuple[str, pd.DataFrame] | None:
    config = FRED_ALTERNATES.get(name)
    if not config:
        return None

    source = config["source"]
    if source == "treasury":
        series = _fetch_treasury_series(config["data"], config["field"], start, end)
    elif source == "cboe":
        series = _fetch_cboe_vix(start, end)
    elif source == "bls":
        series = _fetch_bls_series(config["series_id"], start, end)
    elif source == "manufacturing_growth_proxy":
        series = _fetch_manufacturing_growth_proxy(config["series_id"], start, end)
        source = series.attrs.get("source", source)
    elif source == "ccc_oas_seed":
        series = _fetch_ccc_oas_series(config["ticker"], start, end)
        source = series.attrs.get("source", source)
    else:
        raise RuntimeError(f"Unknown alternate source {source}")

    return source, series.rename(name).to_frame()


def _get_fred_series(
    ticker: str, start: str, end: str | None, retries: int = 1, timeout: int = 12
) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return FredReader(
                ticker,
                start=start,
                end=end,
                retry_count=1,
                pause=1.0,
                timeout=timeout,
            ).read()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def get_fred(
    series_dict: dict[str, str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = ".cache/data",
    refresh: bool = False,
    return_status: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[SourceStatus]]:
    frames = []
    statuses = []
    cache_root = Path(cache_dir)
    for name, ticker in series_dict.items():
        print(f"  FRED {ticker} -> {name}", flush=True)
        path = _cache_path(cache_root, "fred", ticker, start, end)
        try:
            if path.exists() and not refresh:
                s = _read_cached_series(path, ticker)
                statuses.append(
                    SourceStatus(
                        "fred",
                        name,
                        ticker,
                        "cache",
                        len(s),
                        str(path),
                        latest_observation=_latest_observation(s),
                    )
                )
            else:
                s = _get_fred_series(ticker, start, end)
                _write_cached_series(path, s)
                statuses.append(
                    SourceStatus(
                        "fred",
                        name,
                        ticker,
                        "live",
                        len(s),
                        str(path),
                        fetched_at=_now_utc(),
                        latest_observation=_latest_observation(s),
                    )
                )
            s.columns = [name]
            frames.append(s)
        except Exception as exc:
            try:
                alternate = _get_alternate_series(name, start, end)
                if alternate:
                    source, alt = alternate
                    s = alt.rename(columns={name: ticker})
                    _write_cached_series(path, s)
                    statuses.append(
                        SourceStatus(
                            source,
                            name,
                            ticker,
                            "alternate",
                            len(s),
                            str(path),
                            fetched_at=_now_utc(),
                            latest_observation=_latest_observation(s),
                        )
                    )
                    s.columns = [name]
                    frames.append(s)
                    continue
            except Exception as alt_exc:
                alt_error = f"; alternate error: {alt_exc}"
            else:
                alt_error = "; no alternate source configured"

            if path.exists():
                try:
                    s = _read_cached_series(path, ticker)
                    s.columns = [name]
                    frames.append(s)
                    statuses.append(
                        SourceStatus(
                            "fred",
                            name,
                            ticker,
                            "cache_after_error",
                            len(s),
                            str(path),
                            f"{exc}{alt_error}",
                            latest_observation=_latest_observation(s),
                        )
                    )
                    continue
                except Exception as cache_exc:
                    statuses.append(
                        SourceStatus(
                            "fred",
                            name,
                            ticker,
                            "failed",
                            error=f"{exc}{alt_error}; cache error: {cache_exc}",
                        )
                    )
            else:
                statuses.append(
                    SourceStatus(
                        "fred", name, ticker, "failed", error=f"{exc}{alt_error}"
                    )
                )
    if not frames:
        raise RuntimeError(
            "No FRED series downloaded. Check internet connection or FRED availability."
        )
    df = pd.concat(frames, axis=1).sort_index()
    failed = [s for s in statuses if s.status == "failed"]
    if failed:
        print("Warning: some FRED series failed:")
        for status in failed:
            err = status.error or ""
            print(f"  - {status.name} ({status.ticker}): {err[:120]}")
    if return_status:
        return df, statuses
    return df


def get_etf_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = ".cache/data",
    refresh: bool = False,
    return_status: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[SourceStatus]]:
    cache_root = Path(cache_dir)
    yf_cache_dir = cache_root / "yfinance_runtime"
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(yf_cache_dir))

    frames = []
    statuses = []
    for ticker in tickers:
        print(f"  Yahoo {ticker}", flush=True)
        path = _cache_path(cache_root, "yahoo", ticker, start, end)
        last_error = None
        for attempt in range(1, 3):
            try:
                if path.exists() and not refresh:
                    close = _read_cached_series(path, ticker)[ticker]
                    statuses.append(
                        SourceStatus(
                            "yahoo",
                            ticker,
                            ticker,
                            "cache",
                            len(close),
                            str(path),
                            latest_observation=_latest_observation(close),
                        )
                    )
                    frames.append(close.rename(ticker))
                    break

                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                if raw.empty or "Close" not in raw:
                    raise RuntimeError("empty Yahoo Finance response")
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    if ticker not in close:
                        raise RuntimeError(
                            "missing Close column in Yahoo Finance response"
                        )
                    close = close[ticker]
                close = close.rename(ticker)
                _write_cached_series(path, close.to_frame())
                statuses.append(
                    SourceStatus(
                        "yahoo",
                        ticker,
                        ticker,
                        "live",
                        len(close),
                        str(path),
                        fetched_at=_now_utc(),
                        latest_observation=_latest_observation(close),
                    )
                )
                frames.append(close)
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * attempt)
        else:
            if path.exists():
                try:
                    close = _read_cached_series(path, ticker)[ticker]
                    frames.append(close.rename(ticker))
                    statuses.append(
                        SourceStatus(
                            "yahoo",
                            ticker,
                            ticker,
                            "cache_after_error",
                            len(close),
                            str(path),
                            str(last_error),
                            latest_observation=_latest_observation(close),
                        )
                    )
                except Exception as cache_exc:
                    statuses.append(
                        SourceStatus(
                            "yahoo",
                            ticker,
                            ticker,
                            "failed",
                            error=f"{last_error}; cache error: {cache_exc}",
                        )
                    )
            else:
                statuses.append(
                    SourceStatus(
                        "yahoo", ticker, ticker, "failed", error=str(last_error)
                    )
                )

    if not frames:
        raise RuntimeError(
            "No ETF prices downloaded. Check internet connection or ticker availability."
        )

    failed = [s for s in statuses if s.status == "failed"]
    if failed:
        print("Warning: some ETF downloads failed:")
        for status in failed:
            err = status.error or ""
            print(f"  - {status.ticker}: {err[:120]}")

    prices = pd.concat(frames, axis=1).sort_index()
    if return_status:
        return prices, statuses
    return prices
