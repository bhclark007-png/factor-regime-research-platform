# Factor Regime Research Platform

Decision-support system for identifying market regimes, likely outperforming equity factors, confidence levels, credit leadership signals, regime-change risks, and historical analogs.

The project is designed as a long-term research platform for a daily market brief and future portfolio management system. It prioritizes interpretability, reproducibility, and statistical validation over model complexity.

## Current Capabilities

- Public macro and market data ingestion with cache/fallback status reporting.
- Factor return construction from Kenneth French academic factors and tradeable ETF proxies.
- Credit Leadership Score from HY, IG, and CCC spread signals.
- Regime classification and factor probability estimates.
- Regime Stability Score, data-quality confidence adjustment, and regime-break risk monitoring.
- Historical analog engine based on similar macro/credit environments.
- Baseline model validation against previous-winner, factor momentum, logistic regression, decision tree, equal-weight factors, and SPY.
- Canonical `run_result.json` schema used by the dashboard, brief writer, and future modules.
- Reproducible run artifacts in JSON, CSV, and Markdown.
- Streamlit dashboard for desktop and mobile browsers.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your system Python is newer than the data stack supports, use Python 3.11 or 3.12.

## Data Sources

Primary and fallback sources currently include:

- FRED: credit spreads, rates, macro series.
- U.S. Treasury: nominal and real yield curve fallbacks.
- Cboe: VIX history fallback.
- BLS: CPI and payroll fallback.
- Yahoo Finance: ETF price history.
- Local seed file: `data/ccc_oas_history.csv`.
- Optional Trading Economics API: PMI and CCC history when `TRADING_ECONOMICS_API_KEY` is set.
- Kenneth French Data Library: academic monthly factor portfolios used to extend research history before ETF proxy inception.

Factor source modes are explicit:

- `tradeable` is the default and uses ETF proxies for current implementation.
- `academic` uses Kenneth French factors for longer historical validation and research.
- `combined` uses academic history before ETF proxy availability and ETF proxies after they become available.

## How To Run

Run the backend:

```powershell
python run_agent.py
```

Run with academic factor histories:

```powershell
python run_agent.py --factor-source academic
```

Force fresh data:

```powershell
python run_agent.py --refresh-data
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

The dashboard reads:

```text
output/latest/run_result.json
```

The dashboard includes validation and data-health sections. The daily brief intentionally remains concise and does not include validation tables.

## Output Layout

- `output/runs/<run_id>/`: immutable run artifacts.
- `output/latest/`: latest run for dashboard/email consumers.
- `output/run_history.csv`: simple run log.
- `output/latest/run_result.json`: canonical contract.
- `output/latest/daily_brief.md`: concise human-readable brief.

## Roadmap Summary

- Phase 1: data ingestion, factor scoring, daily brief.
- Phase 2: credit leadership, historical analogs, regime stability.
- Phase 3: portfolio tracking and allocation recommendations.
- Phase 4: security selection research.

See [ROADMAP.md](ROADMAP.md) for detail.

## Development Rules

Every major model change must:

- Update [CHANGELOG.md](CHANGELOG.md).
- Update [docs/assumptions.md](docs/assumptions.md) when assumptions or thresholds change.
- Add an ADR under [docs/adr](docs/adr) when architecture changes.

This is not investment advice. It is a research and decision-support platform.
