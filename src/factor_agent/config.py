FRED_SERIES = {
    # Credit
    "hy_oas": "BAMLH0A0HYM2",       # ICE BofA US High Yield OAS
    "ig_oas": "BAMLC0A0CM",         # ICE BofA US Corporate OAS
    "ccc_oas": "BAMLH0A3HYC",       # ICE BofA CCC & Lower HY OAS
    # Rates / policy
    "dgs2": "DGS2",                 # 2Y Treasury
    "dgs10": "DGS10",               # 10Y Treasury
    "fed_funds": "FEDFUNDS",        # Effective fed funds monthly
    "tips10": "DFII10",             # 10Y real yield
    # Volatility
    "vix": "VIXCLS",                # CBOE VIX close
    # Growth / inflation / labor
    "ism_mfg": "NAPM",              # ISM manufacturing PMI
    "claims": "ICSA",               # Initial claims
    "cpi": "CPIAUCSL",              # CPI all urban consumers
    "payrolls": "PAYEMS",           # Nonfarm payrolls
}

ETF_TICKERS = ["SPY", "MTUM", "QUAL", "VLUE", "USMV", "IWM"]

FACTOR_NAMES = {
    "momentum": "MTUM",
    "quality": "QUAL",
    "value": "VLUE",
    "low_vol": "USMV",
    "small_cap": "IWM",
}
